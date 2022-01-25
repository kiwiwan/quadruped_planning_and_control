'''
Adapted from http://underactuated.mit.edu/humanoids.html#example1

Implements paper:
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''
from SRBD import SRBD
from MiniCheetah import MiniCheetah

import notify2
import pdb
import time
from functools import partial
import numpy as np
import pickle
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, ExtractGradient, SnoptSolver, IpoptSolver,
    InitializeAutoDiff, ExtractValue, ExtractGradient,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint, QuaternionEulerIntegrationConstraint,
    PiecewiseQuaternionSlerp, Quaternion, RollPitchYaw, ge, le, InverseKinematics
)

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess()

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))

def gait_optimization(robot_ctor):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    robot = robot_ctor(plant)
    visualizer = ConnectMeshcatVisualizer(builder,
        scene_graph=scene_graph,
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    robot.set_home(plant, plant_context)
    visualizer.load()
    diagram.Publish(context)

    # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('body'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.GetFrameByName('body'), plant.GetFrameByName('LF_FOOT'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.GetFrameByName('body'), plant.GetFrameByName('RF_FOOT'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.GetFrameByName('body'), plant.GetFrameByName('LH_FOOT'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.GetFrameByName('body'), plant.GetFrameByName('RH_FOOT'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # return
    # print(plant.CalcCenterOfMassPositionInWorld(plant_context))


    # body = plant.GetBodyByName("body")
    # spatial_inertial = body.CalcSpatialInertiaInBodyFrame(plant_context)
    # inertial_B = spatial_inertial.CalcRotationalInertia()
    # inertial = spatial_inertial.get_unit_inertia()
    # print(inertial.CopyToFullMatrix3())
    # print(inertial1.CopyToFullMatrix3())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('body'))
    # inertial2 = inertial1.ReExpress(X_WF.rotation())
    # print(inertial2.CopyToFullMatrix3())
    # CalcRotationalInertia()
    # return




    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName(robot.get_body_name())


    PositionView = robot.PositionView()
    VelocityView = robot.VelocityView()

    mu = 1 # rubber on rubber
    total_mass = robot.get_total_mass(context)
    gravity = plant.gravity_field().gravity_vector()
    g = 9.81
    # max normal force assumed to be 4mg
    max_contact_force = 4*g*total_mass

    contact_frame = robot.get_contact_frames()

    in_stance = robot.get_stance_schedule()
    N = robot.get_num_timesteps()
    is_laterally_symmetric = robot.get_laterally_symmetric()
    check_self_collision = robot.get_check_self_collision()
    stride_length = robot.get_stride_length()
    speed = robot.get_speed()

    T = stride_length / speed
    if is_laterally_symmetric:
        T = T / 2.0

    prog = MathematicalProgram()

    # Time steps
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(T/N, T/N, h)
    # prog.AddBoundingBoxConstraint(0.5*T/N, 2.0*T/N, h)
    # prog.AddLinearConstraint(sum(h) >= .9*T)
    # prog.AddLinearConstraint(sum(h) <= 1.1*T)

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    vdot = prog.NewContinuousVariables(nv, N, "vdot")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = robot.get_position_cost()
    v_cost = robot.get_velocity_cost()
    for n in range(N):
        # Joint limits
        # prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:,n])
        # Joint velocity limits
        # prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:,n])
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        # Body orientation
        prog.AddConstraint(OrientationConstraint(plant,
                                                 body_frame, RotationMatrix(),
                                                 plant.world_frame(), RotationMatrix(),
                                                 robot.max_body_rotation(), context[n]), q[:,n])
        # Initial guess for all joint angles is the home position
        # prog.SetInitialGuess(q[:,n], q0)  # Solvers get stuck if the quaternion is initialized with all zeros.

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0]*nv, v[:,n])

    # Make a new autodiff context for this constraint (to maximize cache hits)
    ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1+nq, 1+nq+nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(q, ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])):
                ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
            v_from_qdot = ad_plant.MapQDotToVelocity(ad_velocity_dynamics_context[context_index], (qn - q)/h)
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(context[context_index], (qn - q)/h)
        return v - v_from_qdot
    for n in range(N-1):
        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n),
            lb=[0]*nv, ub=[0]*nv,
            vars=np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1])))
        prog.AddConstraint(eq(v[:,n+1], v[:,n] + h[n]*vdot[:,n]))

    # #add QuaternionEulerIntegrationConstraint
    # for n in range(N-1):
    #     prog.AddConstraint(eq(q[4:,n+1], q[4:,n] + h[n]*v[3:,n]))
    #     prog.AddConstraint(QuaternionEulerIntegrationConstraint(allow_quaternion_negation=False),
    #         np.concatenate((q[:4,n], q[:4,n+1], v[:3,n], [h[n]])))

    # Contact forces
    num_contacts = robot.get_num_contacts()
    '''
    Ordered as follows
    [[[ contact0_x_t0, ... , contact0_x_tn],
      [ contact0_y_t0, ... , contact0_y_tn],
      [ contact0_z_t0, ... , contact0_z_tn]],
     ...
     [[ contactn_x_t0, ... , contactn_x_tn],
      [ contactn_y_t0, ... , contactn_y_tn],
      [ contactn_z_t0, ... , contactn_z_tn]]]
    '''
    normalized_contact_force = [prog.NewContinuousVariables(3, N-1, f"contact{contact}_normalized_contact_force") for contact in range(num_contacts)]
    for n in range(N-1):
        for contact in range(num_contacts):
            # Linear friction cone
            prog.AddLinearConstraint(normalized_contact_force[contact][0,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(-normalized_contact_force[contact][0,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(normalized_contact_force[contact][1,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(-normalized_contact_force[contact][1,n] <= mu*normalized_contact_force[contact][2,n])
            # normal force >=0, normal_force == 0 if not in_stance
            prog.AddBoundingBoxConstraint(0.0, in_stance[contact,n], normalized_contact_force[contact][2,n])

            prog.SetInitialGuess(normalized_contact_force[contact][2,n], 0.25*in_stance[contact,n])




    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, q[4:6,0])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, v[5,0])
    # CoM height
    prog.AddBoundingBoxConstraint(robot.min_com_height(), np.inf, q[6,:])
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, v[3,:])
    # CoM final x position
    if is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(stride_length/2.0, stride_length/2.0, q[4,-1])
    else:
        prog.AddBoundingBoxConstraint(stride_length, stride_length, q[4,-1])
    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddLinearConstraint(eq(total_mass*vdot[3:,n],
            sum(max_contact_force*normalized_contact_force[i][:,n] for i in range(num_contacts)) + total_mass*gravity))

        # prog.AddQuadraticErrorCost(np.diag(np.array([0.1, 0.1, 0.1])), [0]*3, vdot[3:,n])
        # for contact in range(num_contacts):
        #     prog.AddQuadraticErrorCost(np.diag(np.array([0., 0., 0.0000001])), [0]*3, normalized_contact_force[contact][:,n])




    
    foot_p = [prog.NewContinuousVariables(3, N, f"contact{contact}_foot_p") for contact in range(num_contacts)]
    
    #foot order [LF_FOOT, RF_FOOT, LH_FOOT, RH_FOOT]
    xf_nominal_stance = 0.25
    xh_nominal_stance = 0.13
    y_nominal_stance = 0.14
    z_nominal_stance = -0.27
    #foot_nominal_stance in com coordinate
    foot_nominal_stance = np.array([[xf_nominal_stance, y_nominal_stance, z_nominal_stance], 
                                    [xf_nominal_stance, -y_nominal_stance,  z_nominal_stance],
                                    [-xh_nominal_stance,  y_nominal_stance,  z_nominal_stance], 
                                    [-xh_nominal_stance, -y_nominal_stance,  z_nominal_stance]])
    # foot_box = np.array([0.15, 0.1, 0.1]) 
    foot_box = np.array([1.5*0.15, 0.1, 0.1])
    for contact in range(num_contacts):
        if not in_stance[contact, 0]:
            # prog.AddBoundingBoxConstraint(foot_nominal_stance[contact][0]-0.1*abs(foot_nominal_stance[contact][0]), foot_nominal_stance[contact][0]+0.1*abs(foot_nominal_stance[contact][0]), foot_p[contact][0,0])
            prog.AddBoundingBoxConstraint(foot_nominal_stance[contact][1]-0.1*abs(foot_nominal_stance[contact][1]), foot_nominal_stance[contact][1]+0.1*abs(foot_nominal_stance[contact][1]), foot_p[contact][1,0])

    for contact in range(num_contacts):
        prog.AddCost(0.1*((foot_p[contact][0,0] - foot_nominal_stance[contact][0])**2))
        # prog.AddCost(0.1*((foot_p[contact][0,1] - foot_nominal_stance[contact][0])**2))
    for n in range(N):



        # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
        # if n < N-1:# and n > 0
            # active_contacts = np.where(in_stance[:,n])[0]
            # prog.AddConstraint(eq(Hdot[:,n],
            #     sum(np.cross(foot_p[i][:,n] - q[4:,n], max_contact_force*normalized_contact_force[i][:,n]) for i in active_contacts)))


        for contact in range(num_contacts):
            #TODO multiply R convert foot to com coordinate
            # Kinematic range constraints
            prog.AddLinearConstraint(ge(foot_p[contact][:,n]-q[4:,n], foot_nominal_stance[contact]-foot_box))
            prog.AddLinearConstraint(le(foot_p[contact][:,n]-q[4:,n], foot_nominal_stance[contact]+foot_box))
            
            # prog.AddQuadraticErrorCost(np.diag(np.array([0.1, 0.1])), np.array([foot_nominal_stance[contact][1], 0.]), foot_p[contact][1:,n])
            prog.AddQuadraticCost(0.1*((foot_p[contact][2,n])**2))
            prog.AddCost(0.1*((foot_p[contact][1,n] - foot_nominal_stance[contact][1])**2))

   
            if in_stance[contact, n]:
                # Kinematic constraints
                # foot should be on the ground (world position z=0)
                prog.AddBoundingBoxConstraint(0, 0, foot_p[contact][2,n])
                if n > 0 and in_stance[contact, n-1]:
                    # feet should not move during stance.
                    prog.AddLinearConstraint(eq(foot_p[contact][:,n], foot_p[contact][:,n-1]))
                
                # #foot at nominal stance,not consider rotation
                # prog.AddCost((foot_p[contact][:,n] - com[:,n] - foot_nominal_stance[contact]).dot(np.diag(np.array([0.00000001, 4, 0.00000001]) )).dot(foot_p[contact][:,n] - com[:,n] - foot_nominal_stance[contact]))
                # prog.AddLinearConstraint(eq(foot_p[contact][1:2,n], foot_nominal_stance[contact][1:2]))
            else:
                min_clearance = 0.01
                prog.AddBoundingBoxConstraint(min_clearance, np.inf, foot_p[contact][2,n])#np.inf

                # #foot at nominal stance,not consider rotation
                # prog.AddCost((foot_p[contact][:,n] - com[:,n] - foot_nominal_stance[contact]).dot(np.diag(np.array([0.000001, 0.05, 0.01]) )).dot(foot_p[contact][:,n] - com[:,n] - foot_nominal_stance[contact]))
                
                #foot in com can't change too quick
                # if n > 0:
                #     prog.AddCost((foot_p[contact][:,n] - com[:,n] - foot_p[contact][:,n-1] + com[:,n-1]).dot(np.diag(np.array([0.00000001, 5, 0.0000001]) )).dot(foot_p[contact][:,n] - com[:,n] - foot_p[contact][:,n-1] + com[:,n-1]))
            
                if n > 0 and n < N - 1 and not in_stance[contact, n]:
                    # swing foot smooth constrain
                    prog.AddLinearConstraint(eq(foot_p[contact][:2,n], (foot_p[contact][:2,n+1]+foot_p[contact][:2,n-1])/2))
                    # prog.AddLinearConstraint(eq(foot_p[contact][:2,n] - com[:2,n], (foot_p[contact][:2,n+1] - com[:2,n+1] + foot_p[contact][:2,n-1] - com[:2,n-1])/2))


    
    dynamics_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def rotation_dynamics_constraint(vars, context_index, active_contacts):
        q, v, vdot, normalized_contact_force, foot_p = np.split(vars, [nq, nq+nv, nq+2*nv, nq+2*nv+num_contacts*3])

        contact_force = max_contact_force*(normalized_contact_force.reshape(3, num_contacts, order='F'))
        foot_p = foot_p.reshape(3, num_contacts, order='F')
        if isinstance(vars[0], AutoDiffXd):
            #set plant pos and vel
            if not autoDiffArrayEqual(q, ad_plant.GetPositions(dynamics_constraint_context[context_index])):
                ad_plant.SetPositions(dynamics_constraint_context[context_index], q)

            eval_plant = ad_plant
            eval_plant_context = dynamics_constraint_context[context_index]
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)

            eval_plant = plant
            eval_plant_context = context[context_index]

        body = eval_plant.GetBodyByName("body")
        spatial_inertial = body.CalcSpatialInertiaInBodyFrame(eval_plant_context)
        inertial_B = spatial_inertial.CalcRotationalInertia()
        X_WF = eval_plant.CalcRelativeTransform(eval_plant_context, eval_plant.world_frame(), eval_plant.GetFrameByName('body'))
        inertial_W = inertial_B.ReExpress(X_WF.rotation())
        torque_left = np.cross(v[:3],inertial_W.CopyToFullMatrix3().dot(v[:3])) + inertial_W.CopyToFullMatrix3().dot(vdot[:3])
        
        torque = np.zeros(3, dtype='object')
        for contact in active_contacts:
            torque += np.cross(foot_p[:,contact] - q[4:], contact_force[:,contact])

        return torque_left - torque 

    for n in range(N-1):
        active_contacts = np.where(in_stance[:,n])[0]

        Fn = np.concatenate([normalized_contact_force[i][:,n] for i in range(num_contacts)])
        foot_pn = np.concatenate([foot_p[i][:,n] for i in range(num_contacts)])
        prog.AddConstraint(partial(rotation_dynamics_constraint, context_index=n, active_contacts=active_contacts),
            lb=np.zeros(3), ub=np.zeros(3), vars=np.concatenate((q[:,n], v[:,n], vdot[:,n], Fn, foot_pn)))



    # # TODO: Add collision constraints
 
    # Periodicity constraints
    if is_laterally_symmetric:
        robot.add_periodic_constraints(prog, q_view, v_view)
        #other foot constrain need to add 
        # for i in range(num_contacts):
        #     prog.AddLinearEqualityConstraint(foot_p[i][2,0] == foot_p[i][2,-1])
        # for n in range(N):


        prog.AddLinearEqualityConstraint(foot_p[0][0,0] - q[4,0] == foot_p[1][0,-1] - q[4,-1])
        prog.AddLinearEqualityConstraint(foot_p[0][1,0] - q[5,0] == -foot_p[1][1,-1] + q[5,-1])
        # prog.AddLinearEqualityConstraint(foot_p[0][2,0] - q[6,0] == foot_p[1][2,-1] - q[6,-1])
        prog.AddLinearEqualityConstraint(foot_p[0][2,0]== foot_p[1][2,-1])

        prog.AddLinearEqualityConstraint(foot_p[2][0,0] - q[4,0] == foot_p[3][0,-1] - q[4,-1])
        prog.AddLinearEqualityConstraint(foot_p[2][1,0] - q[5,0] == -foot_p[3][1,-1] + q[5,-1])
        # prog.AddLinearEqualityConstraint(foot_p[2][2,0] - q[6,0] == foot_p[3][2,-1] - q[6,-1])
        prog.AddLinearEqualityConstraint(foot_p[2][2,0]== foot_p[3][2,-1])

        prog.AddLinearEqualityConstraint(foot_p[0][0,-1] - q[4,-1] == foot_p[1][0,0] - q[4,0])
        prog.AddLinearEqualityConstraint(foot_p[0][1,-1] - q[5,-1] == -foot_p[1][1,0] + q[5,0])
        # prog.AddLinearEqualityConstraint(foot_p[0][2,-1] - q[6,-1] == foot_p[1][2,0] - q[6,0])
        prog.AddLinearEqualityConstraint(foot_p[0][2,-1]== foot_p[1][2,0])

        prog.AddLinearEqualityConstraint(foot_p[2][0,-1] - q[4,-1] == foot_p[3][0,0] - q[4,0])
        prog.AddLinearEqualityConstraint(foot_p[2][1,-1] - q[5,-1] == -foot_p[3][1,0] + q[5,0])
        # prog.AddLinearEqualityConstraint(foot_p[2][2,-1] - q[6,-1] == foot_p[3][2,0] - q[6,0])
        prog.AddLinearEqualityConstraint(foot_p[2][2,-1]== foot_p[3][2,0])

        # CoM velocity
        prog.AddLinearEqualityConstraint(v[3,0] == v[3,-1])
        prog.AddLinearEqualityConstraint(v[4,0] == -v[4,-1])
        prog.AddLinearEqualityConstraint(v[5,0] == v[5,-1])
    else:
        # Everything except body_x is periodic
        q_selector = robot.get_periodic_view()
        prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1]))
        prog.AddLinearConstraint(eq(v[:,0], v[:,-1]))

        for contact in range(num_contacts):
            prog.AddLinearEqualityConstraint(foot_p[contact][0,0] - q[4,0] == foot_p[contact][0,-1] - q[4,-1])
            prog.AddLinearEqualityConstraint(foot_p[contact][1,0] - q[5,0] == foot_p[contact][1,-1] - q[5,-1])
            prog.AddLinearEqualityConstraint(foot_p[contact][2,0] - q[6,0] == foot_p[contact][2,-1] - q[6,-1])



    #set the initial guess
    init_from_file = False
    # init_from_file = True
    tmpfolder = 'resources/'
    if init_from_file:
        with open(tmpfolder +  'Planner_SRDB1/sol.pkl', 'rb' ) as file:
            h_sol, q_sol, v_sol, normalized_contact_force_sol, foot_p_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol = pickle.load( file )

    qf = np.array(q0)
    # CoM final x position
    if is_laterally_symmetric:
        qf[4] = stride_length/2.0
    else:
        qf[4] = stride_length
    q_pos_init = PiecewisePolynomial.FirstOrderHold([0, T], np.vstack([q0[4:], qf[4:]]).T)
    q_quat_init = PiecewiseQuaternionSlerp([0, T], [Quaternion(q0[:4]), Quaternion(qf[:4])])
    v_v_init = q_pos_init.MakeDerivative()
    v_w_init = q_quat_init.MakeDerivative()
    vdot_v_init = v_v_init.MakeDerivative()
    vdot_w_init = v_w_init.MakeDerivative()

   
    w_delt = np.array([[0.],[0.],[0.]])
    # w_delt = np.array([[0.0000000000001],[0.0000000000001],[0.0000000000001]])
    # w_delt = np.array([[0.00001],[0.00001],[0.00001]])
    if init_from_file:
        prog.SetInitialGuess(h, h_sol)
        prog.SetInitialGuess(q, q_sol)
        prog.SetInitialGuess(v, v_sol)
        prog.SetInitialGuess(vdot, vdot_sol)
        for contact in range(num_contacts):
            prog.SetInitialGuess(normalized_contact_force[contact], normalized_contact_force_sol[contact])
            prog.SetInitialGuess(foot_p[contact], foot_p_sol[contact])
    else:
        for n in range(N):
            prog.SetInitialGuess(q[:,n], np.array([np.hstack((Quaternion(q_quat_init.value(n*T/(N-1))).wxyz(), q_pos_init.value(n*T/(N-1)).flatten()))]).T)
            prog.SetInitialGuess(v[:,n], np.vstack((v_w_init.value(n*T/(N-1))+w_delt, v_v_init.value(n*T/(N-1)))))
            prog.SetInitialGuess(vdot[:,n], np.vstack((vdot_w_init.value(n*T/(N-1)), vdot_v_init.value(n*T/(N-1)))))


            for contact in range(num_contacts):
                #Note: not consider Rotation of torso in world
                prog.SetInitialGuess(foot_p[contact][:,n], (foot_nominal_stance[contact].reshape(3,1) + q_pos_init.value(n*T/(N-1))))

                # prog.SetInitialGuess(h[n], T/(N-1))

    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 2e6)
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 500)
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 4000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    prog.SetSolverOption(snopt, 'Scale option', 2)
    prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    file=open('snopt.out','w')
    file.truncate()

    # TODO a few more costs/constraints from
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m

    now = time.time()
    result = Solve(prog)
    # ipopt = IpoptSolver()
    # result = ipopt.Solve(prog)
    print(f"{time.time() - now}s - {result.get_solver_id().name()}: {result.is_success()}, Cost: {result.get_optimal_cost()}")
    #print(result.is_success())  # We expect this to be false if iterations are limited.
    # if not result.is_success():
    #     print(result.GetInfeasibleConstraintNames(prog))


    #save the solution
    h_sol = result.GetSolution(h)
    q_sol = result.GetSolution(q)
    v_sol = result.GetSolution(v)
    vdot_sol = result.GetSolution(vdot)
    normalized_contact_force_sol = [result.GetSolution(normalized_contact_force[contact]) for contact in range(num_contacts)]
    foot_p_sol = [result.GetSolution(foot_p[contact]) for contact in range(num_contacts)]
    # if result.is_success():
    #     with open(tmpfolder + 'Planner_SRDB1/sol.pkl', 'wb') as file:
    #         pickle.dump( [h_sol, q_sol, v_sol, vdot_sol, normalized_contact_force_sol, foot_p_sol], file )
    if result.is_success():
        gait = robot.get_current_gait()
        with open(tmpfolder + 'Planner_SRDB1/' + gait + '_sol.pkl', 'wb') as file:
            pickle.dump( [h_sol, q_sol, v_sol, vdot_sol, normalized_contact_force_sol, foot_p_sol], file )


      
    # print([prog.GetInitialGuess(foot_p[contact]) for contact in range(num_contacts)])
    # for contact in range(num_contacts):
    #     print(np.array([foot_p_sol[contact][:,n]- com_sol[:,n+1] for n in range(N-1)]))
    
    # print(foot_p_sol)
    animate_trajectory(MiniCheetah, h_sol, q_sol, foot_p_sol, is_laterally_symmetric, stride_length)

    notify2.init("Planner.py")
    notify2.Notification("Planner.py", "Done").show()

def animate_trajectory(robot_ctor, h_sol, q_floating_base_sol, foot_p_sol, is_laterally_symmetric, stride_length):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    robot = robot_ctor(plant,add_ground=True)
    visualizer = ConnectMeshcatVisualizer(builder,
        scene_graph=scene_graph,
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    robot.set_home(plant, plant_context)
    visualizer.load()
    diagram.Publish(context)

    q0 = plant.GetPositions(plant_context)
    PositionView = robot.PositionView()
    VelocityView = robot.VelocityView()
    body_frame = plant.GetFrameByName(robot.get_body_name())
    contact_frame = robot.get_contact_frames()

    # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('body'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())
    # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('LF_FOOT'))
    # print(RollPitchYaw(X_WF.rotation()).vector())
    # print(X_WF.translation())

    q_ik_sol = np.zeros((19,len(h_sol)+1))
    # q_ik_sol[:7,0] = q_floating_base_sol[:,0]
    #could not right
    # q_ik_sol[7:,0] = q0[7:]
    for n in range(len(h_sol)+1):
        q_ik_sol[:7,n] = q_floating_base_sol[:,n]


        ik = InverseKinematics(plant, plant_context)
        ik.AddPositionConstraint(body_frame, [0, 0, 0], plant.world_frame(), q_floating_base_sol[4:,n], q_floating_base_sol[4:,n])
        ik.AddOrientationConstraint(body_frame, RotationMatrix(), plant.world_frame(), RotationMatrix(Quaternion(q_floating_base_sol[:4,n])), 0)
        for i in range(robot.get_num_contacts()):
            ik.AddPositionConstraint(contact_frame[i], [0, 0, 0], plant.world_frame(), foot_p_sol[i][:,n], foot_p_sol[i][:,n])
        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
        result = Solve(ik.prog())
        if not result.is_success():
            print("IK failed!")
        q_ik_sol[7:,n] = result.GetSolution(q)[7:]

        # plant.SetPositions(plant_context, q_ik_sol[:,n+1])
        # print(n, "   &&&")
        # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('LF_FOOT'))
        # print(X_WF.translation())
        # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('RF_FOOT'))

        # print(X_WF.translation())
        # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('LH_FOOT'))

        # print(X_WF.translation())
        # X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('RH_FOOT'))

        # print(X_WF.translation())
        # print("################")

    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],h_sol)))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, q_ik_sol)
    visualizer.start_recording()
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    # T = tf
    for t in np.hstack((np.arange(t0, T, visualizer.draw_period), T)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                qt = robot.HalfStrideToFullStride(qt)
                robot.increment_periodic_view(qt, stride_length/2.0)
            stride = stride // 2
        robot.increment_periodic_view(qt, stride*stride_length)
        plant.SetPositions(plant_context, np.array(qt))

        # plant.SetPositions(plant_context, q_sol.value(t))
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()




# Try them all!  The last two could use a little tuning.
minicheetah_walking_trot = partial(SRBD, gait="walking_trot", add_ground=True)
minicheetah_running_trot = partial(SRBD, gait="running_trot", add_ground=True)
minicheetah_rotary_gallop = partial(SRBD, gait="rotary_gallop", add_ground=True)
minicheetah_bound = partial(SRBD, gait="bound", add_ground=True)

# gait_optimization(minicheetah_walking_trot)
# gait_optimization(minicheetah_running_trot)
# gait_optimization(minicheetah_rotary_gallop)
gait_optimization(minicheetah_bound)

# gait_optimization(partial(Atlas, simplified=True))

while True:
    time.sleep(2)
