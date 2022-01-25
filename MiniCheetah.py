import numpy as np
from pydrake.all import (
        Parser, RigidTransform,
        Box, CoulombFriction
)

from typing import NamedTuple
from Robot import Robot

class MiniCheetah(Robot):
    class JointLimit(NamedTuple):
        effort: float
        lower: float
        upper: float
        velocity: float

    JOINT_LIMITS = {
            "torso_to_abduct_fl_j" : JointLimit(18, -1.6, 1.6, 40),
            "abduct_fl_to_thigh_fl_j" : JointLimit(18, -2.6, 2.6, 40),
            "thigh_fl_to_knee_fl_j" : JointLimit(26, -2.6, 2.6, 26),

            "torso_to_abduct_fr_j" : JointLimit(18, -1.6, 1.6, 40),
            "abduct_fr_to_thigh_fr_j" : JointLimit(18, -2.6, 2.6, 40),
            "thigh_fr_to_knee_fr_j" : JointLimit(26, -2.6, 2.6, 26),

            "torso_to_abduct_hl_j" : JointLimit(18, -1.6, 1.6, 40),
            "abduct_hl_to_thigh_hl_j" : JointLimit(18, -2.6, 2.6, 40),
            "thigh_hl_to_knee_hl_j" : JointLimit(26, -2.6, 2.6, 26),

            "torso_to_abduct_hr_j" : JointLimit(18, -1.6, 1.6, 40),
            "abduct_hr_to_thigh_hr_j" : JointLimit(18, -2.6, 2.6, 40),
            "thigh_hr_to_knee_hr_j" : JointLimit(26, -2.6, 2.6, 26),
    }

    CONTACTS_PER_FRAME = {
            "LF_FOOT": np.array([
                [0, 0, 0,] 
            ]).T,
            "RF_FOOT": np.array([
                [0, 0, 0,] 
            ]).T,
            "LH_FOOT": np.array([
                [0, 0, 0,] 
            ]).T,
            "RH_FOOT": np.array([
                [0, 0, 0,] 
            ]).T
    }

    NUM_ACTUATED_DOF = 12

    def __init__(self, plant, gait="walking_trot", add_ground=False):
        if add_ground:
            color = np.array([.9, .9, .9, 1.0])

            box = Box(30., 30., 1.)
            X_WBox = RigidTransform([0, 0, -0.5-0.0175])

            plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box,
                    "GroundVisuaGeometry", color)

            ground_friction = CoulombFriction(1.0, 1.0)
            plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box,
                    "GroundCollisionGeometry", ground_friction)
            plant.set_penetration_allowance(1.0e-3)
            plant.set_stiction_tolerance(1.0e-3)

        super().__init__(plant, "robots/mini_cheetah/mini_cheetah_mesh.urdf")
        # super().__init__(plant, "robots/mini_cheetah/mini_cheetah_mesh_leg_less_mass.urdf")


        

        self.gait = gait

        # setup gait
        self.is_laterally_symmetric = False
        self.check_self_collision = False
        if gait == 'running_trot':
            self.N = 21
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[1, 3:17] = 1
            self.in_stance[2, 3:17] = 1
            # self.speed = 2.0            #0.95-0.55,1.2-0.75,1.5-0.85,1.5-1.15,1.5-1.65,2.0-1.65,2.5-1.65
            # self.stride_length = 1.65
            self.speed = 2.2
            self.stride_length = 0.65
            self.is_laterally_symmetric = True
        elif gait == 'walking_trot':
            self.N = 21
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, :11] = 1
            self.in_stance[1, 8:self.N] = 1
            self.in_stance[2, 8:self.N] = 1
            self.in_stance[3, :11] = 1
            # self.speed = 1.1
            # self.stride_length = .55
            self.speed = 2.2
            self.stride_length = .6
            self.is_laterally_symmetric = True
        # elif gait == 'walking_trot':
        #     self.N = 41
        #     self.in_stance = np.zeros((4, self.N))
        #     self.in_stance[0, :21] = 1
        #     self.in_stance[1, 16:self.N] = 1
        #     self.in_stance[2, 16:self.N] = 1
        #     self.in_stance[3, :21] = 1
        #     self.speed = 1.1
        #     self.stride_length = .55
        #     self.is_laterally_symmetric = True
        elif gait == 'rotary_gallop':
            self.N = 41
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, 7:19] = 1
            self.in_stance[1, 3:15] = 1
            self.in_stance[2, 24:35] = 1
            self.in_stance[3, 26:38] = 1
            # self.speed = 1.8     #1.6-0.75,
            # self.stride_length = 0.65
            self.speed = 2.2
            self.stride_length = 1.0
            self.check_self_collision = True
        elif gait == 'bound':
            self.N = 41
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, 6:18] = 1
            self.in_stance[1, 6:18] = 1
            self.in_stance[2, 21:32] = 1
            self.in_stance[3, 21:32] = 1
            self.speed = 2.2        #0.65-0.65,1.65-1.2,2.0-1.65
            self.stride_length = 1.2
            self.check_self_collision = True
            # self.is_laterally_symmetric = True
        else:
            raise RuntimeError('Unknown gait.')

    def get_current_gait(self):
        return self.gait

    def get_contact_frames(self):
        return [
            self.plant.GetFrameByName('LF_FOOT'),
            self.plant.GetFrameByName('RF_FOOT'),
            self.plant.GetFrameByName('LH_FOOT'),
            self.plant.GetFrameByName('RH_FOOT')]

    def get_effort_limits(self):
        return [joint_limit[1].effort for joint_limit in self.JOINT_LIMITS.items()]


    def get_contact_frame_names(self):
        return [
            'LF_FOOT',
            'RF_FOOT',
            'LH_FOOT',
            'RH_FOOT']

    def set_home(self, plant, context):
        hip_roll = .1;
        # hip_pitch = 1;
        hip_pitch = 0.5;
        knee = 1.55;
        plant.GetJointByName("torso_to_abduct_fr_j").set_angle(context, -hip_roll)
        plant.GetJointByName("abduct_fr_to_thigh_fr_j").set_angle(context, -hip_pitch)
        plant.GetJointByName("thigh_fr_to_knee_fr_j").set_angle(context, knee)
        plant.GetJointByName("torso_to_abduct_fl_j").set_angle(context, hip_roll)
        plant.GetJointByName("abduct_fl_to_thigh_fl_j").set_angle(context, -hip_pitch)
        plant.GetJointByName("thigh_fl_to_knee_fl_j").set_angle(context, knee)
        plant.GetJointByName("torso_to_abduct_hr_j").set_angle(context, -hip_roll)
        plant.GetJointByName("abduct_hr_to_thigh_hr_j").set_angle(context, -hip_pitch)
        plant.GetJointByName("thigh_hr_to_knee_hr_j").set_angle(context, knee)
        plant.GetJointByName("torso_to_abduct_hl_j").set_angle(context, hip_roll)
        plant.GetJointByName("abduct_hl_to_thigh_hl_j").set_angle(context, -hip_pitch)
        plant.GetJointByName("thigh_hl_to_knee_hl_j").set_angle(context, knee)
        plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.27]))  #0.24984  +0.0175  0.270375

    def get_stance_schedule(self):
        return self.in_stance

    def get_num_timesteps(self):
        return self.N

    def get_laterally_symmetric(self):
        return self.is_laterally_symmetric

    def get_check_self_collision(self):
        return self.check_self_collision

    def get_stride_length(self):
        return self.stride_length

    def get_speed(self):
        return self.speed

    def get_body_name(self):
        return "body"

    def max_body_rotation(self):
        return 0.1

    def min_com_height(self):
        return 0.15  #0.125

#Centroid
    # def get_position_cost(self):
    #     q_cost = self.PositionView()([1]*self.nq)
    #     q_cost.body_x = 0
    #     q_cost.body_y = 0
    #     q_cost.body_qx = 0
    #     q_cost.body_qy = 0
    #     q_cost.body_qz = 0
    #     q_cost.body_qw = 0
    #     q_cost.torso_to_abduct_fl_j = 5
    #     q_cost.torso_to_abduct_fr_j = 5
    #     q_cost.torso_to_abduct_hl_j = 5
    #     q_cost.torso_to_abduct_hr_j = 5
    #     return q_cost

    # def get_velocity_cost(self):
    #     v_cost = self.VelocityView()([1]*self.nv)
    #     v_cost.body_vx = 0
    #     v_cost.body_wx = 0
    #     v_cost.body_wy = 0
    #     v_cost.body_wz = 0
    #     return v_cost

# #Centroid_new
#     def get_position_cost(self):
#         q_cost = self.PositionView()([1]*self.nq)
#         q_cost.body_x = 0
#         q_cost.body_y = 0
#         q_cost.body_z = 0
#         q_cost.body_qx = 0
#         q_cost.body_qy = 0
#         q_cost.body_qz = 0
#         q_cost.body_qw = 0
#         # q_cost.torso_to_abduct_fl_j = 5
#         # q_cost.torso_to_abduct_fr_j = 5
#         # q_cost.torso_to_abduct_hl_j = 5
#         # q_cost.torso_to_abduct_hr_j = 5
#         return q_cost

#     def get_velocity_cost(self):
#         v_cost = self.VelocityView()([0.00000001]*self.nv)
#         v_cost.body_vx = 0
#         v_cost.body_vy = 0
#         v_cost.body_vz = 0
#         v_cost.body_wx = 0
#         v_cost.body_wy = 0
#         v_cost.body_wz = 0
#         return v_cost

#Centroid_new+torque
    def get_position_cost(self):
        q_cost = self.PositionView()([0.5]*self.nq)
        q_cost.body_x = 0
        q_cost.body_y = 0
        q_cost.body_z = 0
        q_cost.body_qx = 0
        q_cost.body_qy = 0
        q_cost.body_qz = 0
        q_cost.body_qw = 0
        # q_cost.torso_to_abduct_fl_j = 5
        # q_cost.torso_to_abduct_fr_j = 5
        # q_cost.torso_to_abduct_hl_j = 5
        # q_cost.torso_to_abduct_hr_j = 5
        return q_cost

    def get_velocity_cost(self):
        v_cost = self.VelocityView()([0.00000001]*self.nv)
        v_cost.body_vx = 0
        v_cost.body_vy = 0
        v_cost.body_vz = 0
        v_cost.body_wx = 0
        v_cost.body_wy = 0
        v_cost.body_wz = 0
        return v_cost

    # def get_position_cost(self):
    #     q_cost = self.PositionView()([0.0000001]*self.nq)
    #     q_cost.body_x = 0
    #     q_cost.body_y = 0
    #     q_cost.body_z = 0
    #     q_cost.body_qx = 0
    #     q_cost.body_qy = 0
    #     q_cost.body_qz = 0
    #     q_cost.body_qw = 0
    #     q_cost.torso_to_abduct_fl_j = 0.00001
    #     q_cost.torso_to_abduct_fr_j = 0.00001
    #     q_cost.torso_to_abduct_hl_j = 0.00001
    #     q_cost.torso_to_abduct_hr_j = 0.00001
    #     return q_cost

    # def get_velocity_cost(self):
    #     v_cost = self.VelocityView()([0.00000001]*self.nv)
    #     v_cost.body_vx = 0
    #     v_cost.body_vy = 0
    #     v_cost.body_vz = 0
    #     v_cost.body_wx = 0
    #     v_cost.body_wy = 0
    #     v_cost.body_wz = 0
    #     return v_cost

# #Centroid_SRBD_real+torque
#     def get_position_cost(self):
#         q_cost = self.PositionView()([1]*self.nq)
#         q_cost.body_x = 0
#         q_cost.body_y = 0
#         q_cost.body_z = 0
#         q_cost.body_qx = 0
#         q_cost.body_qy = 0
#         q_cost.body_qz = 0
#         q_cost.body_qw = 0
#         q_cost.torso_to_abduct_fl_j = 5
#         q_cost.torso_to_abduct_fr_j = 5
#         q_cost.torso_to_abduct_hl_j = 5
#         q_cost.torso_to_abduct_hr_j = 5
#         return q_cost

#     def get_velocity_cost(self):
#         v_cost = self.VelocityView()([0.00000001]*self.nv)
#         v_cost.body_vx = 0
#         v_cost.body_vy = 0
#         v_cost.body_vz = 0
#         v_cost.body_wx = 0
#         v_cost.body_wy = 0
#         v_cost.body_wz = 0
#         return v_cost

# #Centroid_SRBD_real
#     def get_position_cost(self):
#         q_cost = self.PositionView()([1]*self.nq)
#         q_cost.body_x = 0
#         q_cost.body_y = 0
#         q_cost.body_z = 0
#         q_cost.body_qx = 0
#         q_cost.body_qy = 0
#         q_cost.body_qz = 0
#         q_cost.body_qw = 0
#         # q_cost.torso_to_abduct_fl_j = 5
#         # q_cost.torso_to_abduct_fr_j = 5
#         # q_cost.torso_to_abduct_hl_j = 5
#         # q_cost.torso_to_abduct_hr_j = 5
#         return q_cost

#     def get_velocity_cost(self):
#         v_cost = self.VelocityView()([0.00000001]*self.nv)
#         v_cost.body_vx = 0
#         v_cost.body_vy = 0
#         v_cost.body_vz = 0
#         v_cost.body_wx = 0
#         v_cost.body_wy = 0
#         v_cost.body_wz = 0
#         return v_cost

#Centroid_SRBD
    # def get_position_cost(self):
    #     q_cost = self.PositionView()([0.00001]*self.nq)
    #     q_cost.body_x = 0
    #     q_cost.body_y = 0
    #     q_cost.body_z = 0
    #     q_cost.body_qx = 0
    #     q_cost.body_qy = 0
    #     q_cost.body_qz = 0
    #     q_cost.body_qw = 0
    #     q_cost.torso_to_abduct_fl_j = 0.00001
    #     q_cost.torso_to_abduct_fr_j = 0.00001
    #     q_cost.torso_to_abduct_hl_j = 0.00001
    #     q_cost.torso_to_abduct_hr_j = 0.00001
    #     return q_cost

    # def get_velocity_cost(self):
    #     v_cost = self.VelocityView()([0.0001]*self.nv)
    #     v_cost.body_vx = 0
    #     v_cost.body_vy = 0
    #     v_cost.body_vz = 0
    #     v_cost.body_wx = 0
    #     v_cost.body_wy = 0
    #     v_cost.body_wz = 0
    #     return v_cost


    def get_periodic_view(self):
        q_selector = self.PositionView()([True]*self.nq)
        q_selector.body_x = False
        return q_selector

    def increment_periodic_view(self, view, increment):
        view.body_x += increment

    def add_periodic_constraints(self, prog, q_view, v_view):
        # Joints
        def AddAntiSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        def AddSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])

        AddAntiSymmetricPair(q_view.torso_to_abduct_fl_j,
                             q_view.torso_to_abduct_fr_j)
        AddSymmetricPair(q_view.abduct_fl_to_thigh_fl_j,
                         q_view.abduct_fr_to_thigh_fr_j)
        AddSymmetricPair(q_view.thigh_fl_to_knee_fl_j, q_view.thigh_fr_to_knee_fr_j)
        AddAntiSymmetricPair(q_view.torso_to_abduct_hl_j,
                             q_view.torso_to_abduct_hr_j)
        AddSymmetricPair(q_view.abduct_hl_to_thigh_hl_j,
                         q_view.abduct_hr_to_thigh_hr_j)
        AddSymmetricPair(q_view.thigh_hl_to_knee_hl_j, q_view.thigh_hr_to_knee_hr_j)
        prog.AddLinearEqualityConstraint(q_view.body_y[0] == -q_view.body_y[-1])
        prog.AddLinearEqualityConstraint(q_view.body_z[0] == q_view.body_z[-1])
        # Body orientation must be in the xz plane:
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qz[[0,-1]])

        # Floating base velocity
        prog.AddLinearEqualityConstraint(v_view.body_vx[0] == v_view.body_vx[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vy[0] == -v_view.body_vy[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vz[0] == v_view.body_vz[-1])

    def HalfStrideToFullStride(self, a):
        b = self.PositionView()(np.copy(a))

        b.body_y = -a.body_y
        # Mirror quaternion so that roll=-roll, yaw=-yaw
        b.body_qx = -a.body_qx
        b.body_qz = -a.body_qz

        b.torso_to_abduct_fl_j = -a.torso_to_abduct_fr_j
        b.torso_to_abduct_fr_j = -a.torso_to_abduct_fl_j
        b.torso_to_abduct_hl_j = -a.torso_to_abduct_hr_j
        b.torso_to_abduct_hr_j = -a.torso_to_abduct_hl_j

        b.abduct_fl_to_thigh_fl_j = a.abduct_fr_to_thigh_fr_j
        b.abduct_fr_to_thigh_fr_j = a.abduct_fl_to_thigh_fl_j
        b.abduct_hl_to_thigh_hl_j = a.abduct_hr_to_thigh_hr_j
        b.abduct_hr_to_thigh_hr_j = a.abduct_hl_to_thigh_hl_j

        b.thigh_fl_to_knee_fl_j = a.thigh_fr_to_knee_fr_j
        b.thigh_fr_to_knee_fr_j = a.thigh_fl_to_knee_fl_j
        b.thigh_hl_to_knee_hl_j = a.thigh_hr_to_knee_hr_j
        b.thigh_hr_to_knee_hr_j = a.thigh_hl_to_knee_hl_j

        return b

def getAllJointIndicesInGeneralizedPositions(plant):
    for joint_limit in MiniCheetah.JOINT_LIMITS.items():
        index = getJointIndexInGeneralizedPositions(plant, joint_limit[0])
        print(f"{joint_limit[0]}: {index}")

def getActuatorIndex(plant, joint_name):
    return int(plant.GetJointActuatorByName(joint_name).index())

def getJointLimitsSortedByActuator(plant):
    return sorted(MiniCheetah.JOINT_LIMITS.items(), key=lambda entry : getActuatorIndex(plant, entry[0]))

def getJointIndexInGeneralizedPositions(plant, joint_name):
    return int(plant.GetJointByName(joint_name).position_start())

def getJointIndexInGeneralizedVelocities(plant, joint_name):
    return getJointIndexInGeneralizedPositions(plant, joint_name) - 1

'''
Returns the joint limits sorted by their position in the generalized positions
'''
def getJointLimitsSortedByPosition(plant):
    return sorted(MiniCheetah.JOINT_LIMITS.items(),
        key=lambda entry : getJointIndexInGeneralizedPositions(plant, entry[0]))

def getJointValues(plant, joint_names, context):
    ret = []
    for name in joint_names:
        ret.append(plant.GetJointByName(name).get_angle(context))
    return ret


def setJointValues(plant, joint_values, context):
    for i in range(len(joint_values)):
        plant.GetJointByIndex(i).set_angle(context, joint_values[i])            

