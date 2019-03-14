import sys, csv, pathlib, math
import numpy as np
import numpy.random as rnd
from scipy.interpolate import splprep, splev, spalde
import pyctcr
""" Remember to run to following command in the terminal before use of this script:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mni/Documents/pyctr_forward_model/lib/release """

F_MIN_DEFAULT = 0.0
F_MAX_DEFAULT = 0.5
TUBE_LENGTH = np.array((.501, .337, .180))  # default cannula values
CANNULA_PATH = "../../ctcr_configs/cannula.xml"

#F_MAX_DEFAULT = 0.05
#TUBE_LENGTH = np.array((.370, .305, .170))  # grassmann2018neural lengths
#CANNULA_PATH = "../../ctcr_configs/grassmann_cannula.xml"


def write_csv(data, path="csv", filename="dataset"):
    name = str(filename)
    pathlib.Path(path).mkdir(exist_ok=True)

    with open(str(path) + str(name) + ".csv", "w+", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        for row in data:
            writer.writerow(row)


def count_zero_rows(data):
    n_rows = np.shape(data)[0]
    n_columns = np.shape(data)[1]
    n_zero_rows = 0

    """iterate from the bottom of the matrix toward the top, counting the 
    number of 0-rows. As soon as a non-zero row is encountered, break."""
    for i in range((n_rows - 1), 0, -1):
        if (data[i] == np.zeros((1, n_columns))).all():  # if all elements == 0
            n_zero_rows += 1
        else:
            break
    return n_zero_rows


def compute_screw(frenet_frame, position, ref_frenet_frame, ref_position):  #TODO
    pass


class Dataset:  # adventurous_ant

    def __init__(self, f_min = F_MIN_DEFAULT, f_max = F_MAX_DEFAULT, f_z_max=F_MAX_DEFAULT , n_samples = 100, n_sample_attempts = 200, n_spacecurve_points = 30, allow_negative_forces=True, verbose=True, csv_path="csv/", csv_filename="dataset"):
        """:param f_max maximum magnitude of force vector in Newtons"""


        self.model = pyctcr.CTCR_Model(CANNULA_PATH)
        self.f_min = f_min
        self.f_max = f_max  # the CTR model has failed with forces as low as 0.071 N, 0.11 N works most of the time though
        self.f_z_max = f_z_max  # this variable limits the z-component |F_z| < f_z_max
        self.n_sample_target = n_samples
        self.n_sample_attempts = n_sample_attempts
        self.n_samples = n_sample_attempts
        self.allow_negative_vectors = allow_negative_forces;
        self.verbose = verbose
        self.csv_path = csv_path
        self.csv_filename = csv_filename

        """if n_spacecurve_points>0 the curve is interpolated and resampled to n_spacecurve_points points, otherwise, 
        the curve is not resampled, and the number of points may vary, depending on the pyctr outputs"""
        self.n_spacecurve_points = n_spacecurve_points

        # Randomly generate matrics of alpha, beta, & force values"""
        self.alpha = self.generate_alpha()  # alpha_rad:    [-pi, pi]
        self.beta = self.generate_beta()  #    beta_m:    [-0.501, 0.0]; [-0.337, 0.0]; [-.180, 0.0]
        self.force = self.generate_force()  # |force_N|:    [0., f_max]

        self.poses = np.zeros((n_sample_attempts, self.n_spacecurve_points, 3))
        self.compliances = np.zeros((self.n_sample_attempts, 6, 6))
        self.tip_orientations = np.zeros((self.n_sample_attempts, 9))
        #self.poses_unforced = np.zeros((n_sample_attempts, self.n_spacecurve_points, 3))
        self.data_mask = np.ones(n_sample_attempts, dtype=bool)  # must be intialized to 1

        """:param training_data matrix where each row consists of(alpha, beta, xyz[30])"""
        self.training_data = np.zeros((self.n_samples, (6 + 3 * self.n_spacecurve_points)))

        self.generate_dataset()

    def print_attribute_shapes(self):
        print("alpha", np.shape(self.alpha))
        print("beta", np.shape(self.beta))
        print("force", np.shape(self.force))
        print("poses", np.shape(self.poses))
        print("data_mask", np.shape(self.data_mask))

    def resample_ctr(self, pose):
        pose_T = np.transpose(pose)
        tck, u = splprep([pose_T[0], pose_T[1], pose_T[2]], s=0)

        uniform = []
        for i in range(0, self.n_spacecurve_points):
            uniform.append(i / (self.n_spacecurve_points - 1.))
        u = np.array(uniform)

        new_points = splev(u, tck)
        #derivatives = spalde(u, tck)  # TODO: take a look at these derivatives, and use them to calculate the Frenet frames
        resampled_pts = np.reshape(new_points, (3, self.n_spacecurve_points)).transpose()
        return resampled_pts

    def generate_alpha(self, lower_bound=-np.pi, upper_bound=np.pi):
        return rnd.uniform(lower_bound, upper_bound, (self.n_sample_attempts, 3))

    def generate_beta(self):
        """ Stochastic sampling to create betas """
        betas = np.zeros((self.n_sample_attempts, 3))
        for i in range(0, self.n_sample_attempts):
            valid = False
            while not valid:
                c_random = rnd.uniform(-1.0, 0.0, (3))
                b = c_random * TUBE_LENGTH
                start_0 = b[0]
                start_1 = b[1]
                start_2 = b[2]
                end_0 = b[0] + TUBE_LENGTH[0]
                end_1 = b[1] + TUBE_LENGTH[1]
                end_2 = b[2] + TUBE_LENGTH[2]

                if (start_0 <= start_1) & (start_1 <= start_2) & (end_0 >= end_1) & (end_1 >= end_2):
                    betas[i] = b
                    valid = True
        return betas

    def generate_force(self):
        """ generate force vectors with directions uniformly sampled from a unit sphere. [n_samples, 3] """
        """ Note that randomly generating vectors in the interval [-1, 1]^3 and unitizing them, introduces a bias towards
        the corners of the cube. To get uniformly sampled unit vectors, they must be drawn from a sphere (i.e. the
        vectors in the corners, with radius > 1, must be discarded. Alternatively, lib boost uniform_on_sphere could be
        used. See: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space """

        force = np.zeros((self.n_sample_attempts, 3))
        vector_count = 0

        if self.allow_negative_vectors:
            direction_vector_lower_bound = -1.
        else:  # force vectors lie in the positive octant only
            direction_vector_lower_bound = 0.

        # generate magnitudes
        magnitude = rnd.uniform(self.f_min, self.f_max, self.n_sample_attempts)

        """ For performance reasons the random vectors are batch-generated. Since some of the vectors (the excess vectors 
        in the corners) are discarded, multiple passes may be required. Each pass tries n_sample_attempts times."""
        # vector_count is the current number of successful vectors (i.e. vectors of length <= 1)
        while vector_count < self.n_sample_attempts:
            vector_pool = rnd.uniform(direction_vector_lower_bound, 1., (self.n_sample_attempts, 3))  # batch-generate random vectors
            for v in vector_pool:
                r_squared = sum(v * v)
                if r_squared <= 1.:  # select only vectors of radius <= 1
                    unitized_direction_vector = v * 1/(r_squared ** 0.5)  # vector*scalar multiplication
                    f = unitized_direction_vector * magnitude[vector_count]
                    if abs(f[2]) <= self.f_z_max:
                        force[vector_count] = f
                        vector_count += 1
                    else:
                        pass  # discard this vector because z-component is too large
                else:
                    pass  # discard this vector
                if vector_count == self.n_sample_attempts:
                    break  # out of the for-loop

        return force

    def prune_faulty_samples(self):
        # prune all faulty entries out
        self.alpha = self.alpha[self.data_mask]
        self.beta = self.beta[self.data_mask]
        self.force = self.force[self.data_mask]
        self.poses = self.poses[self.data_mask]
        self.compliances = self.compliances[self.data_mask]
        self.tip_orientations = self.tip_orientations[self.data_mask]

        self.n_samples = np.sum(self.data_mask)

        print("{} / {} samples successful.".format(self.n_samples, self.n_sample_attempts))

        if self.n_samples < self.n_sample_target:
            raise RuntimeError("Generation of dataset failed. Too many failed PYCTCR samples!")
        elif self.n_samples > self.n_sample_target:  # if the dataset is oversized, cut it down to the desired size
            self.alpha = self.alpha[:self.n_sample_target]
            self.beta = self.beta[:self.n_sample_target]
            self.force = self.force[:self.n_sample_target]
            self.poses = self.poses[:self.n_sample_target]
            self.compliances = self.compliances[:self.n_sample_target]
            self.tip_orientations = self.tip_orientations[:self.n_sample_target]

            self.n_samples = self.n_sample_target
            print("Pruning down to {} samples.".format(self.n_samples))

        #self.data_mask = np.ones(self.n_samples, dtype=bool) # Commented out, so that the mask remains, in case I have to prune other arrays in subclasses

    def pyctcr_calculation(self, alpha=np.zeros(3), beta=np.zeros(3), force=np.zeros(3), filter_and_resample=True):
        """Compute pose from rotation alpha (+/-pi), translations beta, & wrench (F, M)"""
        # this data can be handed over for plotting directly
        # TODO: I could hash the (alpha,beta,force) values to a 6-8 digit hex value, to give them a unique identifier for plotting.

        wrench = np.concatenate((force, np.zeros(3)))
        self.model._wrench = wrench
        self.model.move(alpha, beta)
        pose = self.model.poses

        if np.count_nonzero(pose) == 0:
            raise ValueError("All pose coordinates are zero. PYCTR failed.\n{}".format(parameter_string))

        n_zero_rows = count_zero_rows(pose)
        if filter_and_resample:  # resample the curve
            pose_filtered = pose[:-n_zero_rows]
            pose = self.resample_ctr(pose_filtered)

        return pose, self.model.orientation[-(n_zero_rows+1)], self.model.compliance

    def compute_poses(self, force):  # TODO: compute 0-force poses here
        """compute a whole dataset for training the ELM from matrices of inputs"""

        poses = np.zeros((self.n_sample_attempts, self.n_spacecurve_points, 3))
        tip_orientations = np.zeros((self.n_sample_attempts, 9))
        compliances = np.zeros((self.n_sample_attempts, 6, 6))

        print("Computing {} samples...".format(self.n_sample_attempts))
        for i in range(0, self.n_sample_attempts):
            if self.verbose:
                if i % (self.n_sample_attempts/10) == 0:  # print whenever another nth of the data has been completed
                    print("{:>5}%... ".format(int(100 * i / self.n_sample_attempts)), end='', flush=True)
            try:
                poses[i], tip_orientations[i], compliances[i] = self.pyctcr_calculation(self.alpha[i], self.beta[i], force[i])
                # mask stays set to true (default)
            except:
                self.data_mask[i] = False
                """ initializing to TRUE and setting false in case of a failure allows multiple passes of compute_poses
                over the inputs, with different forces for instance, and guarantees that if any one pass fails, the mask
                 is set appropriately to filter out the corresponding samples. """
            #except ctr.PyCTCRError:  # TODO figure out what type of exception the PYCTR throws when the simulation has too many steps
            #    self.data_mask[i] = False
        print("\n")
        return poses, tip_orientations, compliances

    def compose_dataset(self):
        dim_input = 6 + 3 * self.n_spacecurve_points  # dimension of final input vector of the dataset
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.poses[i].flatten()), axis=0)
        print("force, alpha, beta, pose_xyz")
        return data

    def generate_dataset(self):
        """Generate a dataset for the ELM"""
        self.poses, self.tip_orientations, self.compliances = self.compute_poses(self.force)
        self.prune_faulty_samples()
        self.training_data = self.compose_dataset()
        """ training_data is: [alpha, beta, pose-xyz]"""

    def write_csv_files(self):
        print("Writing {} samples to CSV files.".format(self.n_samples))
        # self.write_csv("alpha_beta", np.concatenate((self.alpha,self.beta), axis=1))
        #write_csv("inputs", self.training_data)
        #write_csv("force", self.force)
        write_csv(np.concatenate((self.force, self.training_data), axis=1), path=str(self.csv_path), filename=str(self.csv_filename))
        #write_csv("poses", np.reshape(self.poses, (self.n_samples, 3 * self.n_spacecurve_points)))  # for plotting


class FixedAlphaDataset(Dataset):
    def generate_alpha(self):
        """ set rotation angles alpha to 0 """
        return np.zeros((self.n_sample_attempts, 3))


class FixedBetaDataset(Dataset):
    def generate_beta(self, c0_min=-1.0, c0_max=0.0):
        """ Set all beta to 0, i.e. tubes fully extended """
        return np.zeros((self.n_sample_attempts, 3))


class FixedAlphaBeta(FixedAlphaDataset, FixedBetaDataset):  # bigger_beetle
    pass


class GrassmannAlpha(Dataset):
    def generate_alpha(self, lower_bound=-np.pi, upper_bound=np.pi):
        return Dataset.generate_alpha(self, -np.pi/3, np.pi/3)


class GrassmannBeta(Dataset):
    def generate_beta(self):
        """ Stochastic sampling to create betas """
        # for it to really be the same configuration, you have to use the actual tube lengths from the paper:
        assert (TUBE_LENGTH[0]==.370 and TUBE_LENGTH[1]==.305 and TUBE_LENGTH[2] == .170)

        betas = np.zeros((self.n_sample_attempts, 3))
        for i in range(0, self.n_sample_attempts):
            valid = False
            while not valid:
                c_random = rnd.uniform(-1.0, 0.0, (3))
                if c_random[0]*TUBE_LENGTH[0] < -0.144 or c_random[1]*TUBE_LENGTH[1] < -0.115 or c_random[2]*TUBE_LENGTH[2] < -0.081:
                    """ c is the retraction-coefficient of the tubes. c0_min and c0_max constrain the innermost 
                    (longest) tube, and thus the overall length of the robot. """
                    continue
                b = c_random * TUBE_LENGTH
                start_0 = b[0]
                start_1 = b[1]
                start_2 = b[2]
                end_0 = b[0] + TUBE_LENGTH[0]
                end_1 = b[1] + TUBE_LENGTH[1]
                end_2 = b[2] + TUBE_LENGTH[2]

                if (start_0 <= start_1) & (start_1 <= start_2) & (end_0 >= end_1) & (end_1 >= end_2):
                    betas[i] = b
                    valid = True
        return betas


class RestrictedBeta(Dataset):
    def generate_beta(self):
        """ Stochastic sampling to create betas """
        betas = np.zeros((self.n_sample_attempts, 3))
        for i in range(0, self.n_sample_attempts):
            valid = False
            while not valid:
                c_random = rnd.uniform(-0.35, 0.0, (3))
                b = c_random * TUBE_LENGTH
                start_0 = b[0]
                start_1 = b[1]
                start_2 = b[2]
                end_0 = b[0] + TUBE_LENGTH[0]
                end_1 = b[1] + TUBE_LENGTH[1]
                end_2 = b[2] + TUBE_LENGTH[2]

                if (start_0 <= start_1) & (start_1 <= start_2) & (end_0 >= end_1) & (end_1 >= end_2):
                    betas[i] = b
                    valid = True
        return betas


class GrassmannAlphaBeta(GrassmannAlpha, GrassmannBeta):
    pass


class RestrictedAlphaBeta(GrassmannAlpha, RestrictedBeta):
    pass


class XYForceDataset(Dataset):
    """ generate forces in the XY-plane without any Z-component"""

    def generate_force(self):
        direction_vectors = np.zeros((self.n_sample_attempts, 2))
        unitized_direction_vectors = np.zeros((self.n_sample_attempts, 2))
        force = np.zeros((self.n_sample_attempts, 2))
        radius_squared = np.zeros(self.n_sample_attempts)
        vector_count = 0

        """ For performance reasons the random vectors are batch-generated. Since some of the vectors (the excess vectors 
        in the corners) are discarded, multiple passes may be required. Each pass tries n_sample_attempts times."""

        if self.allow_negative_vectors:
            direction_vector_lower_bound = -1.
        else:  # force vectors lie in the positive octant only
            direction_vector_lower_bound = 0.

        # vector_count is the current number of successful vectors (i.e. vectors of length <= 1)
        while vector_count < self.n_sample_attempts:
            vector_pool = rnd.uniform(direction_vector_lower_bound, 1., (self.n_sample_attempts, 2))  # batch-generate random vectors
            for v in vector_pool:
                r_squared = sum(v * v)
                if r_squared <= 1.:  # select only vectors of radius <= 1
                    direction_vectors[vector_count] = v
                    radius_squared[vector_count] = r_squared
                    vector_count += 1
                else:  # discard this vector
                    pass
                if vector_count == self.n_sample_attempts:
                    break  # out of the for-loop

        # unitize all vectors in direction_vectors
        for i in range(0, self.n_sample_attempts):
            unitized_direction_vectors[i] = direction_vectors[i] * 1/(radius_squared[i]**0.5)

        # generate magnitudes
        magnitude = rnd.uniform(0., self.f_max, self.n_sample_attempts)

        for i in range(0, self.n_sample_attempts):
            force[i] = unitized_direction_vectors[i] * magnitude[i]

        force = np.concatenate((force, np.zeros((self.n_sample_attempts,1))), axis=1)
        assert (np.shape(force) == (self.n_sample_attempts, 3))  # verify that the concatenation is correct

        return force


class ReferenceDataset(Dataset):
    def generate_dataset(self):
        self.poses, self.tip_orientations, self.compliances = self.compute_poses(self.force)
        print("Generate 0-force reference poses:")
        self.poses_unforced, self.tip_orientations_unforced, self.compliances_unforced = self.compute_poses(np.zeros((self.n_sample_attempts, 3)))
        self.prune_faulty_samples()
        self.training_data = self.compose_dataset()

    def compose_dataset(self):
        dim_input = 6 + 3 * self.n_spacecurve_points  # dimension of final input vector of the dataset
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.poses_unforced[i].flatten(), self.poses[i].flatten()), axis=0)
        print("force, alpha, beta, pose_xyz_unforced, pose_xyz")
        return data


class EndPointRefDataset(ReferenceDataset):
    """ Tip coordiantes only, for unforced and forced configuration. orientation info not included."""

    def prune_faulty_samples(self):
        self.poses_unforced = self.poses_unforced[self.data_mask]
        self.tip_orientations_unforced = self.tip_orientations_unforced[self.data_mask]
        self.compliances_unforced = self.compliances_unforced[self.data_mask]

        # if the dataset is oversized, cut it down to the desired size
        if self.n_samples > self.n_sample_target:
            self.poses_unforced = self.poses_unforced[:self.n_sample_target]
            self.tip_orientations_unforced = self.tip_orientations_unforced[:self.n_sample_target]
            self.compliances_unforced = self.compliances_unforced[:self.n_sample_target]

        Dataset.prune_faulty_samples(self)

    def compose_dataset(self):
        dim_input = (6 + 3 * 2)
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.poses_unforced[i].flatten()[-3:], self.poses[i].flatten()[-3:]), axis=0)
        print("force, alpha, beta, tip_xyz_unforced, tip_xyz")
        return data


class EndPointDataset(Dataset):
    def compose_dataset(self):
        dim_input = 6 + 3
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.poses[i].flatten()[-3:]), axis=0)
        print("force, alpha, beta, tip_xyz")
        return data


class DeltaPoseDataset(Dataset):

    def generate_dataset(self):
        self.poses, self.tip_orientations, self.compliance = self.compute_poses(self.force)
        poses_unforced, o, c = self.compute_poses(np.zeros((self.n_sample_attempts, 3)))
        self.poses = np.subtract(poses_unforced, self.poses)
        self.prune_faulty_samples()
        self.training_data = self.compose_dataset()


class FullyExtendedEndPtDataset(FixedAlphaBeta, EndPointRefDataset):  # capable_caterpillar
    pass  # do nothing


class TipDeltaDataset(FixedAlphaBeta, DeltaPoseDataset, EndPointDataset):  # daring_dragonfly
    """computes the delta-pose and filters down to (force, alpha, beta, delta-tip-coords)"""
    pass


class NoisyDataset(Dataset):

    def make_noisy_alpha(self, mean=0.0, stdev_deg=5):
        noise = np.random.normal(mean, math.radians(stdev_deg), (np.shape(self.alpha)))
        self.alpha = self.alpha + noise

    def make_noisy_beta(self, mean=0.0, stdev=.005):
        noise = np.random.normal(mean, stdev, (np.shape(self.beta)))
        self.beta = self.beta + noise

    def make_noisy_poses(self, mean=0.0, stdev=.003):
        noise = np.random.normal(mean, stdev, (np.shape(self.poses)))
        self.poses = self.poses + noise

    def make_noise(self):
        self.make_noisy_poses()

    def generate_dataset(self):
        """Generate a dataset for the ELM"""
        self.poses, self.tip_orientations, self.compliances = self.compute_poses(self.force)
        self.prune_faulty_samples()
        self.make_noise()
        self.training_data = self.compose_dataset()
        """ training_data is: [alpha, beta, pose-xyz]"""


class NoisyPoseFixedAlphaBeta(FixedAlphaBeta, NoisyDataset):
    pass


class NoisyPoseXYDataset(FixedAlphaBeta, NoisyDataset, XYForceDataset):
    pass


class NoisyXYRandomAlphaDataset(FixedBetaDataset, XYForceDataset, NoisyDataset):
    pass


class ForceComplianceDataset(FixedAlphaBeta):
    def compose_dataset(self):
        dim_input = 6 + 9 + 3*self.n_spacecurve_points
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.compliances[i, 0:3, 0:3].flatten(), self.poses[i].flatten()), axis=0)
        return data


class TipOrientationRefDataset(EndPointRefDataset):
    def compose_dataset(self):
        dim_input = (6 + (3+9)*2)
        data = np.zeros((self.n_samples, dim_input))
        for i in range(0, self.n_samples):
            data[i] = np.concatenate(
                (self.alpha[i], self.beta[i],
                 self.poses_unforced[i].flatten()[-3:], self.tip_orientations_unforced[i],
                 self.poses[i].flatten()[-3:], self.tip_orientations[i]), axis=0)
        print("Note that this dataset uses the 2nd-last orientation as a replacement for the real tip orientation (which is 0).\n")
        print("force, alpha, beta, tip_xyz_unforced, tip_orientation_unforced, tip_xyz, tip_orientation")
        return data


class TipOrientationRefFixedXY(TipOrientationRefDataset, FixedAlphaBeta, XYForceDataset):
    pass


class TipOrientationRefRestrictedXY(TipOrientationRefDataset, RestrictedAlphaBeta, XYForceDataset):
    pass


class TipOrientationRefFullrandomXY(TipOrientationRefDataset, XYForceDataset):
    pass


class TipOrientationRefFixedXYZ(TipOrientationRefDataset, FixedAlphaBeta):
    pass


class TipOrientationRefRestrictedXYZ(TipOrientationRefDataset, RestrictedAlphaBeta):  # THE STANDARD DATASET
    pass


class TipOrientationRefFullrandomXYZ(TipOrientationRefDataset):
    pass


class OrientationDataset(Dataset):
    def compose_dataset(self):
        dim_input = 6 + 9 + 3 * self.n_spacecurve_points  # dimension of final input vector of the dataset
        data = np.zeros((self.n_samples, dim_input))

        for i in range(0, self.n_samples):
            data[i] = np.concatenate((self.alpha[i], self.beta[i], self.tip_orientations[i], self.poses[i].flatten()), axis=0)
        print("force, alpha, beta, tip_orientation, pose_xyz")
        return data


class OrientationRestrictedXY(RestrictedAlphaBeta, XYForceDataset, OrientationDataset):
    pass


class PiecewiseLinearOrientationRefCompliance(ReferenceDataset):

    def compute_piecewise_linear(self, pose):  # pose should be (n_points * 3)
        result = np.zeros((self.n_spacecurve_points-1, 3))
        for i in range(0, self.n_spacecurve_points-1):
            result[i] = pose[i+1] - pose[i]
        return result

    def compose_dataset(self):
        dim_input = 6 + ((self.n_spacecurve_points-1)*3+9)*2 + 36
        data = np.zeros((self.n_samples, dim_input))

        pose_piecewise_unforced = np.zeros((self.n_samples, self.n_spacecurve_points-1, 3))
        pose_piecewise = np.zeros((self.n_samples, self.n_spacecurve_points-1, 3))

        for i in range(0, self.n_samples):
            pose_piecewise_unforced[i] = self.compute_piecewise_linear(self.poses_unforced[i])
            pose_piecewise[i] = self.compute_piecewise_linear(self.poses[i])

        for i in range(0, self.n_samples):
            data[i] = np.concatenate(
                (self.alpha[i], self.beta[i],
                 pose_piecewise_unforced[i].flatten(), self.tip_orientations_unforced[i],
                 pose_piecewise[i].flatten(), self.tip_orientations[i],
                 self.compliances_unforced[i].flatten()), axis=0)
        print("Note that this dataset uses the 2nd-last orientation as a replacement for the real tip orientation (which is 0).\n")
        print("force, alpha, beta, vectors_unforced, tip_orientation_unforced, vectors, tip_orientation, compliance_unforced")
        return data


# TODO: this code doesn't work, there must be some problem wtih the inputs
# class AntiHysteresisDataset(Dataset):
#     def pyctcr_calculation(self, alpha=np.zeros(3), beta=np.zeros(3), force=np.zeros(3), filter_and_resample=True):
#         """Compute pose from rotation alpha (+/-pi), translations beta, & wrench (F, M)"""
#
#         wrench = np.concatenate((force, np.zeros(3)))
#         self.model.release_tip_wrench()
#         self.model.move(np.zeros(3), np.zeros(3))
#         self.model.move(alpha, beta)
#         self.model.apply_tip_wrench(wrench)
#         pose = self.model.poses
#
#         if np.count_nonzero(pose) == 0:
#             raise ValueError("All pose coordinates are zero. PYCTR failed.\n{}".format(parameter_string))
#
#         n_zero_rows = count_zero_rows(pose)
#         if filter_and_resample:  # resample the curve
#             pose_filtered = pose[:-n_zero_rows]
#             pose = self.resample_ctr(pose_filtered)
#
#         return pose, self.model.orientation[-(n_zero_rows+1)], self.model.compliance
#
#
# class AntiHysteresisStandard(TipOrientationRefDataset, RestrictedAlphaBeta, AntiHysteresisDataset):
#     pass


def pyctcrSampleCall(n_samples):
    ds = Dataset(f_max=0.5, n_samples=n_samples, n_sample_attempts=(4 * n_samples), verbose=False)
    return ds.training_data


def fixedAlphaBetaSampleCall(n_samples, f_max):
    ds = FixedAlphaBeta(f_max=f_max, n_samples=n_samples, n_sample_attempts=int(3 * n_samples), verbose=False)
    #print("force[0] = " + str(ds.force[0]))
    return ds.training_data


def tipDataSampleCall(n_samples):
    """matlab entry point for use with FullyExtendedEndPtDataset"""
    ds = FullyExtendedEndPtDataset(n_samples=n_samples, n_sample_attempts=int(2*n_samples), verbose=False)
    return ds.training_data


def deltaTipSampleCall(n_samples):
    """matlab entry point for use with TipDeltaDataset"""
    ds = TipDeltaDataset(n_samples=n_samples, n_sample_attempts=int(1.1*n_samples), verbose=False)
    return ds.training_data


def forceComplianceDatasetSampleCall(n_samples, f_max):
    ds = ForceComplianceDataset(f_max=f_max, n_samples=n_samples, n_sample_attempts=int(3 * n_samples), verbose=False)
    return ds.training_data


if __name__ == "__main__":  # if this file is being run as a script:
    """ Usage examples as a standalone script:
            python3 dataset.py 
            python3 dataset.py 10
    """
    n_samples_default = 100

    if len(sys.argv) >= 2:
        if int(sys.argv[1]) > 0:
            n_samples_default = int(sys.argv[1])
        if len(sys.argv) > 2:
            raise Exception("Too many command line arguments.")


    #ds = DeltaPoseDataset(n_samples=n_samples_default, n_sample_attempts=(2 * n_samples_default))
    #ds = TipDeltaDataset(n_samples=n_samples_default, n_sample_attempts=(int(1.1 * n_samples_default)))  #daring_dragonfly fmax=0.05
    #ds = FixedAlphaBeta(n_samples=n_samples_default,n_sample_attempts=(int(3 * n_samples_default)), csv_path="csv/", csv_filename="FixedAlphaBeta")
    #ds = ForceComplianceDataset(n_samples=n_samples_default,n_sample_attempts=(int(3 * n_samples_default)))
    #ds = NoisyPoseFixedAlphaBeta(n_samples=n_samples_default,n_sample_attempts=(int(3 * n_samples_default)), csv_path="csv/", csv_filename="NoisyPoseFixedAlphaBeta")
    #ds = NoisyPoseXYDataset(n_samples=n_samples_default,n_sample_attempts=(int(3.4 * n_samples_default)), csv_path="csv/", csv_filename="NoisyPoseXYDataset")

    # Positive octant force dataset:
    #ds = NoisyPoseXYDataset(n_samples=n_samples_default,n_sample_attempts=(int(3.4 * n_samples_default)), allow_negative_forces=False, csv_path="csv/", csv_filename="PositiveOctantNoisyXYDataset")

    #ds = NoisyXYRandomAlphaDataset(n_samples=n_samples_default,n_sample_attempts=(int(3.5 * n_samples_default)), allow_negative_forces=False, csv_path="csv/", csv_filename="NoisyXYRandomAlphaDataset")
    #ds = TipOrientationRefRestrictedXY(n_samples=n_samples_default, n_sample_attempts=(int(3.5 * n_samples_default)), allow_negative_forces=True, csv_path="csv/", csv_filename="TipOrientationRefRestrictedXY")
    #ds = RestrictedAlphaBeta(n_samples=n_samples_default, n_sample_attempts=(int(3.5 * n_samples_default)), allow_negative_forces=True, csv_path="csv/", csv_filename="RestrictedAlphaBeta")

    # ds = TipOrientationRefFixedXYZ(n_samples=n_samples_default, n_sample_attempts=(int(3.0 * n_samples_default)), allow_negative_forces=True, csv_path="csv/", csv_filename="TipOrientationRefFixedXYZ")
    ds = TipOrientationRefRestrictedXYZ(n_samples=n_samples_default, n_sample_attempts=(int(2.0 * n_samples_default)), allow_negative_forces=True, csv_path="csv/", csv_filename="TipOrientationRefRestrictedXYZ")
    #ds = TipOrientationRefFullrandomXYZ(n_samples=n_samples_default, n_sample_attempts=(int(1.5 * n_samples_default)), allow_negative_forces=True, csv_path="csv/", csv_filename="TipOrientationRefFullrandomXYZ")
    

    ds.write_csv_files()



"""
Approximate conversion from N to gram-force. (Factor 10 instead of the actual 9.81)

10    N = 1000 gf
 1    N =  100 gf
 0.1  N =   10 gf
 0.01 N =    1 gf
 
9.8    N = 1000 gf
0.98   N =  100 gf
0.098  N =   10 gf
0.0098 N =    1 gf
"""
