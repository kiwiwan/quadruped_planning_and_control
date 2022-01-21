'''
Adapted from http://underactuated.mit.edu/humanoids.html#example1

Implements paper:
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''
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
    MathematicalProgram, Solve, eq, AutoDiffXd, ExtractGradient, SnoptSolver,
    InitializeAutoDiff, ExtractValue, ExtractGradient,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint, QuaternionEulerIntegrationConstraint,
    PiecewiseQuaternionSlerp, Quaternion, InverseKinematics, RollPitchYaw, ComPositionConstraint
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
    # prog.AddBoundingBoxConstraint(T/N, T/N, h)
    prog.AddBoundingBoxConstraint(0.5*T/N, 2.0*T/N, h)
    prog.AddLinearConstraint(sum(h) >= .9*T)
    prog.AddLinearConstraint(sum(h) <= 1.1*T)

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = robot.get_position_cost()
    v_cost = robot.get_velocity_cost()
    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:,n])
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:,n])
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


    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    # CoM height
    prog.AddBoundingBoxConstraint(robot.min_com_height(), np.inf, com[2,:])
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:])
    # CoM final x position
    if is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(stride_length/2.0, stride_length/2.0, com[0,-1])
    else:
        prog.AddBoundingBoxConstraint(stride_length, stride_length, com[0,-1])
    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n]))
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n]))
        prog.AddConstraint(eq(total_mass*comddot[:,n],
            sum(max_contact_force*normalized_contact_force[i][:,n] for i in range(num_contacts)) + total_mass*gravity))

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index, active_contacts):
        q, com, Hdot, normalized_contact_force = np.split(vars, [nq, nq+3, nq+6])
        '''
        [[ contact0_x_tn, ... , contactn_x_tn ],
         [ contact0_y_tn, ... , contactn_y_tn ],
         [ contact0_z_tn, ... , contactn_z_tn ]]
        '''
        contact_force = max_contact_force*(normalized_contact_force.reshape(3, num_contacts, order='F'))
        if isinstance(vars[0], AutoDiffXd):
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3, dtype='object')
            for contact in active_contacts:
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[contact], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    contact_frame[contact], [0, 0, 0], plant.world_frame(), plant.world_frame())
                ad_p_WF = InitializeAutoDiff(p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                torque += np.cross(ad_p_WF.reshape(3) - com, contact_force[:,contact])
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for contact in active_contacts:
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[contact], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,contact])
        return Hdot - torque
    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))
        active_contacts = np.where(in_stance[:,n])[0]
        '''
        [ contact0_x_tn, contact0_y_tn, contact0_z_tn, ... , contactn_x_tn, contactn_y_tn, contactn_z_tn ]
        '''
        Fn = np.concatenate([normalized_contact_force[i][:,n] for i in range(num_contacts)])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n, active_contacts=active_contacts),
                lb=np.zeros(3), ub=np.zeros(3),
                vars=np.concatenate((q[:,n], com[:,n], Hdot[:,n], Fn)))

    # com == CenterOfMass(q), H = SpatialMomentumInWorldAboutPoint(q, v, com)
    # Make a new autodiff context for this constraint (to maximize cache hits)
    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq+nv, nq+nv+3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(qv, ad_plant.GetPositionsAndVelocities(com_constraint_context[context_index])):
                ad_plant.SetPositionsAndVelocities(com_constraint_context[context_index], qv)
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(com_constraint_context[context_index])
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(com_constraint_context[context_index], com).rotational()
        else:
            if not np.array_equal(qv, plant.GetPositionsAndVelocities(context[context_index])):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(context[context_index])
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(context[context_index], com).rotational()
        return np.concatenate((com_q - com, H_qv - H))
    for n in range(N):
        prog.AddConstraint(partial(com_constraint, context_index=n),
            lb=np.zeros(6), ub=np.zeros(6), vars=np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n])))

    # TODO: Add collision constraints

    # Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(context[context_index], frame, [0,0,0], plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index+1], frame, [0,0,0], plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(context[context_index], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(context[context_index+1], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            return InitializeAutoDiff(
                p_WF_n - p_WF, J_WF_n @ ExtractGradient(qn) - J_WF @ ExtractGradient(q))
        else:
            return p_WF_n - p_WF

    # Kinematic constraints
    def swing_constraint(vars, context_index, frame):
        q_pre, q, qn = np.split(vars, [nq, nq*2])
        if not np.array_equal(q_pre, plant.GetPositions(context[context_index-1])):
            plant.SetPositions(context[context_index-1], q_pre)
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF_pre = plant.CalcPointsPositions(context[context_index-1], frame, [0,0,0], plant.world_frame())
        p_WF = plant.CalcPointsPositions(context[context_index], frame, [0,0,0], plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index+1], frame, [0,0,0], plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF_pre = plant.CalcJacobianTranslationalVelocity(context[context_index-1], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF = plant.CalcJacobianTranslationalVelocity(context[context_index], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(context[context_index+1], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            return InitializeAutoDiff(
                p_WF - (p_WF_pre + p_WF_n)/2, J_WF @ ExtractGradient(q) - (J_WF_pre @ ExtractGradient(q_pre) + J_WF_n @ ExtractGradient(qn))/2)[:1]
        else:
            return p_WF - (p_WF_pre + p_WF_n)/2
    for i in range(robot.get_num_contacts()):
        for n in range(N):
            if in_stance[i, n]:
                # foot should be on the ground (world position z=0)
                prog.AddConstraint(PositionConstraint(
                    plant, plant.world_frame(), [-np.inf,-np.inf,0], [np.inf,np.inf,0],
                    contact_frame[i], [0,0,0], context[n]), q[:,n])
                if n > 0 and in_stance[i, n-1]:
                    # feet should not move during stance.
                    prog.AddConstraint(partial(fixed_position_constraint, context_index=n-1, frame=contact_frame[i]),
                                       lb=np.zeros(3), ub=np.zeros(3), vars=np.concatenate((q[:,n-1], q[:,n])))
            else:
                min_clearance = 0.01
                prog.AddConstraint(PositionConstraint(plant, plant.world_frame(), [-np.inf,-np.inf,min_clearance], [np.inf,np.inf,np.inf],contact_frame[i],[0,0,0],context[n]), q[:,n])
                # if n > 0 and n < N - 1 and not in_stance[i, n]:
                #     # feet should not move during stance.
                #     prog.AddConstraint(partial(swing_constraint, context_index=n, frame=contact_frame[i]),
                #                        lb=np.zeros(1), ub=np.zeros(1), vars=np.concatenate((q[:,n-1], q[:,n], q[:,n+1])))
            

    # Periodicity constraints
    if is_laterally_symmetric:
        robot.add_periodic_constraints(prog, q_view, v_view)

        # CoM velocity
        prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1])
        prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1])
        prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1])
    else:
        # Everything except body_x is periodic
        q_selector = robot.get_periodic_view()
        prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1]))
        prog.AddLinearConstraint(eq(v[:,0], v[:,-1]))


    #set the initial guess
    # init_from_file = False
    init_from_file = True
    init_from_SRBD = True
    # init_from_SRBD = False
    tmpfolder = 'resources/'
    if init_from_file:
        if not init_from_SRBD:
            with open(tmpfolder +  'Planner_Cheetah_QEI/sol.pkl', 'rb' ) as file:
                h_sol, q_sol, v_sol, normalized_contact_force_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol = pickle.load( file )
        else:   
            gait = robot.get_current_gait()
            with open(tmpfolder +  'Planner_SRDB/' + gait + '_sol.pkl', 'rb' ) as file:
                h_sol, q_floating_base_sol, v_floating_base_sol, normalized_contact_force_sol, foot_p_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol = pickle.load( file )

            # with open(tmpfolder +  'Planner_SRDB1/' + gait + '_sol.pkl', 'rb' ) as file:
            #     h_sol, q_floating_base_sol, v_floating_base_sol, vdot_floating_base_sol, normalized_contact_force_sol, foot_p_sol = pickle.load( file )


            q_sol = np.zeros((19,len(h_sol)+1))
            for n in range(len(h_sol)+1):
                q_sol[:7,n] = q_floating_base_sol[:,n]


                ik = InverseKinematics(plant, plant_context)
                ik.AddPositionConstraint(body_frame, [0, 0, 0], plant.world_frame(), q_floating_base_sol[4:,n], q_floating_base_sol[4:,n])
                ik.AddOrientationConstraint(body_frame, RotationMatrix(), plant.world_frame(), RotationMatrix(Quaternion(q_floating_base_sol[:4,n])), 0)
                
                for i in range(robot.get_num_contacts()):
                    ik.AddPositionConstraint(contact_frame[i], [0, 0, 0], plant.world_frame(), foot_p_sol[i][:,n], foot_p_sol[i][:,n])
                prog_ik = ik.get_mutable_prog()
                q_ik = ik.q()
                # prog_ik.AddConstraint(ComPositionConstraint(plant, None, plant.world_frame(),plant_context), q_ik)
                prog_ik.AddQuadraticErrorCost(np.identity(len(q_ik)), q0, q_ik)
                prog_ik.SetInitialGuess(q_ik, q0)
                result_ik = Solve(ik.prog())
                if not result_ik.is_success():
                    print("IK failed!")
                q_sol[7:,n] = result_ik.GetSolution(q_ik)[7:]

            v_sol = np.zeros((18,len(h_sol)+1))
            v_sol[:6,:] = v_floating_base_sol[:,:]
            for n in range(N-1):
                v_sol[6:,n+1] = (q_sol[7:,n+1]-q_sol[7:,n])/h_sol[n]

            # com_sol = q_floating_base_sol[4:,:]
            # comdot_sol = v_floating_base_sol[3:,:]
            # comddot_sol = vdot_floating_base_sol[3:,:-1]

            # com_sol = np.zeros((3, N))
            # comdot_sol = np.zeros((3, N))
            # comddot_sol = np.zeros((3, N-1))
            # H_sol = np.zeros((3, N))
            # Hdot_sol = np.zeros((3, N-1))
            # for n in range(N):
            #     plant.SetPositions(context[n], q_sol[:,n])
            #     plant.SetVelocities(context[n], v_sol[:,n])
            #     com_q = plant.CalcCenterOfMassPositionInWorld(context[n])
            #     H_sol[:,n] = plant.CalcSpatialMomentumInWorldAboutPoint(context[n], com_q).rotational()
            #     com_sol[:,n] = com_q

            #     if n < N-1:
            #         active_contacts = np.where(in_stance[:,n])[0]
            #         torque = np.zeros(3)
            #         for contact in active_contacts:
            #             p_WF = plant.CalcPointsPositions(context[n], contact_frame[contact], [0,0,0], plant.world_frame())
            #             torque += np.cross(p_WF.reshape(3) - com_q, max_contact_force*normalized_contact_force_sol[contact][:,n])
            #         Hdot_sol[:,n] = torque
           
            
            # for n in range(N-1):
            #     comddot_sol[:,n] = (sum(max_contact_force*normalized_contact_force_sol[i][:,n] for i in range(num_contacts)) + total_mass*gravity)/total_mass
            # for n in range(N-1):
            #     # Hdot_sol[:,n] = (H_sol[:,n+1]-H_sol[:,n])/h_sol[n]
            #     comdot_sol[:,n+1] = (com_sol[:,n+1]-com_sol[:,n])/h_sol[n]
            #     # comddot_sol[:,n] = (comdot_sol[:,n+1]-comdot_sol[:,n])/h_sol[n]

    qf = np.array(q0)
    if is_laterally_symmetric:
        qf[4] = stride_length/2.0
    else:
        qf[4] = stride_length

    q_pos_init = PiecewisePolynomial.FirstOrderHold([0, T], np.vstack([q0[4:], qf[4:]]).T)
    q_quat_init = PiecewiseQuaternionSlerp([0, T], [Quaternion(q0[:4]), Quaternion(qf[:4])])
    v_v_init = q_pos_init.MakeDerivative()
    v_w_init = q_quat_init.MakeDerivative()

    plant.SetPositions(plant_context, q0)
    com_q0 = plant.CalcCenterOfMassPositionInWorld(plant_context)
    plant.SetPositions(plant_context, qf)
    com_qf = plant.CalcCenterOfMassPositionInWorld(plant_context)
    com_init = PiecewisePolynomial.FirstOrderHold([0, T], np.vstack([com_q0, com_qf]).T)
    comdot_init = com_init.MakeDerivative()
    comddot_init = comdot_init.MakeDerivative()

    w_delt = np.array([[0.],[0.],[0.]])
    # w_delt = np.array([[0.0001],[0.0001],[0.0001]])
    if init_from_file:
        # print("h")
        prog.SetInitialGuess(h, h_sol)
        prog.SetInitialGuess(q, q_sol)
        # # for n in range(N-1):
        # #     prog.SetInitialGuess(v[:,n], np.vstack((w_delt, v_v_init.value(n*T/(N-1)))))
        prog.SetInitialGuess(v, v_sol)
        prog.SetInitialGuess(com, com_sol)
        prog.SetInitialGuess(comdot, comdot_sol)
        prog.SetInitialGuess(comddot, comddot_sol)
        prog.SetInitialGuess(H, H_sol)
        prog.SetInitialGuess(Hdot, Hdot_sol)
        for contact in range(num_contacts):
            prog.SetInitialGuess(normalized_contact_force[contact], normalized_contact_force_sol[contact])
        
        
    else:
        prog.SetInitialGuess(H, np.zeros((3, N)))
        prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
        for n in range(N):
            prog.SetInitialGuess(q[:,n], np.array([np.hstack((Quaternion(q_quat_init.value(n*T/(N-1))).wxyz(), q_pos_init.value(n*T/(N-1)).flatten()))]).T)
            prog.SetInitialGuess(v[:,n], np.vstack((v_w_init.value(n*T/(N-1))+w_delt, v_v_init.value(n*T/(N-1)))))
            if n != N-1:
                prog.SetInitialGuess(com[:,n], com_init.value(n*T/(N-1)))
                prog.SetInitialGuess(comdot[:,n], comdot_init.value(n*T/(N-1)))
                prog.SetInitialGuess(comddot[:,n], comddot_init.value(n*T/(N-1)))

                # prog.SetInitialGuess(h[n], T/(N-1))

                for contact in range(num_contacts):
                    prog.SetInitialGuess(normalized_contact_force[contact][2,n], 0.25*in_stance[contact,n])

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
    print(f"{time.time() - now}s - {result.get_solver_id().name()}: {result.is_success()}, Cost: {result.get_optimal_cost()}")
    #print(result.is_success())  # We expect this to be false if iterations are limited.

    #save the solution
    h_sol = result.GetSolution(h)
    q_sol = result.GetSolution(q)
    v_sol = result.GetSolution(v)
    normalized_contact_force_sol = [result.GetSolution(normalized_contact_force[contact]) for contact in range(num_contacts)]
    com_sol = result.GetSolution(com)
    comdot_sol = result.GetSolution(comdot)
    comddot_sol = result.GetSolution(comddot)
    H_sol = result.GetSolution(H)
    Hdot_sol = result.GetSolution(Hdot)
    if result.is_success():
        with open(tmpfolder + 'Planner_Cheetah_QEI/sol.pkl', 'wb') as file:
            pickle.dump( [h_sol, q_sol, v_sol, normalized_contact_force_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol], file )

    # for n in range(len(h_sol)):
    #     plant.SetPositions(plant_context, q_sol[:,n+1])
    #     print(n, "   &&&")
    #     X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('LF_FOOT'))
    #     print(X_WF.translation())
    #     X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('RF_FOOT'))

    #     print(X_WF.translation())
    #     X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('LH_FOOT'))

    #     print(X_WF.translation())
    #     X_WF = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('RH_FOOT'))

    #     print(X_WF.translation())
    #     print("################")

    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.start_recording()
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
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
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()

    notify2.init("Planner.py")
    notify2.Notification("Planner.py", "Done").show()

    # if not result.is_success():
    #     print(result.GetInfeasibleConstraintNames(prog))
        # pdb.set_trace()

# Try them all!  The last two could use a little tuning.
minicheetah_walking_trot = partial(MiniCheetah, gait="walking_trot")
minicheetah_running_trot = partial(MiniCheetah, gait="running_trot")
minicheetah_rotary_gallop = partial(MiniCheetah, gait="rotary_gallop")
minicheetah_bound = partial(MiniCheetah, gait="bound")

gait_optimization(minicheetah_walking_trot)
# gait_optimization(minicheetah_running_trot)
# gait_optimization(minicheetah_rotary_gallop)
# gait_optimization(minicheetah_bound)

# gait_optimization(partial(Atlas, simplified=True))

while True:
    time.sleep(2)
