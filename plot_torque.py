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
    PiecewiseQuaternionSlerp, Quaternion
)

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess()




def plot_torque(robot_ctor):

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


    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName(robot.get_body_name())
    nq = plant.num_positions()
    nv = plant.num_velocities()
    

    mu = 1 # rubber on rubber
    total_mass = robot.get_total_mass(context)
    gravity = plant.gravity_field().gravity_vector()
    g = 9.81
    # max normal force assumed to be 4mg
    max_contact_force = 4*g*total_mass
    contact_frame = robot.get_contact_frames()

    N = robot.get_num_timesteps()
    
    # tmpfolder = 'resources/'+  'Planner_Cheetah_QEI/'
    tmpfolder = 'resources/'+  'Planner_Cheetah_QEI_SRBD/'
    gait = robot.get_current_gait()  
    # with open(tmpfolder + gait +  '_sol.pkl', 'rb' ) as file:
    #     h_sol, q_sol, v_sol, normalized_contact_force_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol = pickle.load( file )

    with open(tmpfolder +  gait + '_torque' + '_sol.pkl', 'rb' ) as file:
        h_sol, q_sol, v_sol, normalized_contact_force_sol, com_sol, comdot_sol, comddot_sol, H_sol, Hdot_sol = pickle.load( file )


    contact_force_sol = np.array((normalized_contact_force_sol))
    for n in range(N-1):
        for contact in range(robot.get_num_contacts()):
            contact_force_sol[contact][:,n] = max_contact_force*normalized_contact_force_sol[contact][:,n]
    t_sol = np.cumsum(np.concatenate(([0],h_sol)))
    vdot_sol = np.zeros((nv, N))
    for n in range(N-1):
        vdot_sol[:,n] = (v_sol[:,n+1]-v_sol[:,n])/h_sol[n]
    vdot_sol[:,N-1] = vdot_sol[:,N-2]

    tau = []
    for n in range(len(t_sol)-1):

        # qv = np.concatenate((q_poly[0], v_poly[0]))
        qv = np.concatenate((q_sol[:,n], v_sol[:,n]))
        plant.SetPositionsAndVelocities(plant_context, qv)

        H = plant.CalcMassMatrixViaInverseDynamics(plant_context)
        C = plant.CalcBiasTerm(plant_context) - plant.CalcGravityGeneralizedForces(plant_context)
        G = plant.CalcGravityGeneralizedForces(plant_context)
        B = plant.MakeActuationMatrix()
        # print('G: ',G)
        # print('C: ',C)


        v_idx_act = 6 # Start index of actuated joints in generalized velocities
        H_f = H[0:v_idx_act,:]
        H_a = H[v_idx_act:,:]
        C_f = C[0:v_idx_act]
        C_a = C[v_idx_act:]
        B_a = B[v_idx_act:,:]
        # B_a_inv = np.linalg.inv(B_a)

        torque = np.zeros(nv)
        for i in range(robot.get_num_contacts()):
            J_WF = plant.CalcJacobianTranslationalVelocity(plant_context, JacobianWrtVariable.kV,
                                                    contact_frame[i], [0, 0, 0], plant.world_frame(), plant.world_frame())

            torque = torque + J_WF.T.dot(contact_force_sol[i][:,n])
        

        # print('C: ',torque[v_idx_act:])
        # torque = B_a_inv.dot(H_a.dot(vdot_sol[:,n]) + C_a - torque[v_idx_act:])
        torque = H_a.dot(vdot_sol[:,n]) + C_a - torque[v_idx_act:]
        # print('C: ',H_a.dot(vdot_sol[:,n]))

        tau.append(torque)
        if n == N-2:
            tau.append(torque)

        # print('tau ',tau)
    tau = np.array(tau)
    # print('tau: ',tau)

    import matplotlib.pyplot as plt
    plt.ion()
    
    joint = ['hip_roll', 'hip_pitch', 'knee_pitch']
    sub = [311, 312, 313]
    limit = np.vstack((np.array([18]*N),np.array([18]*N),np.array([26]*N)))

    fig = plt.figure(figsize=(5, 10))

    for i in range(3):
        ax = plt.subplot(sub[i])
        plt.plot(t_sol , tau[:,i+3*0], 'o-')
        plt.plot(t_sol , tau[:,i+3*1], 'o-')
        plt.plot(t_sol , tau[:,i+3*2], 'o-')
        plt.plot(t_sol , tau[:,i+3*3], 'o-')
        plt.plot(t_sol , limit[i],'k')
        plt.plot(t_sol , -limit[i],'k')
        if i == 1:
            plt.legend(['LF', 'RF', 'LH', 'RH', 'limit'], loc='lower left', bbox_to_anchor=(1,0))  
        plt.xlabel('Time(s)')
        plt.ylabel(joint[i] + ' torque(Nm)')
        # ax.set_title(joint[i] + ' torque')
        plt.margins(0.01,0.03)


    # fig.legend(['LF', 'RF', 'LH', 'RH', 'limit'], loc='center right', bbox_to_anchor=(0,1))#
    # savfolder = 'plot/'+  'Planner_Cheetah_QEI/'
    savfolder = 'plot/'+  'Planner_Cheetah_QEI_SRBD/'
    plt.ioff()
    # fig.subplots_adjust(right=0.7)
    
    # plt.savefig(savfolder + gait +'.png', bbox_inches='tight')
    plt.savefig(savfolder + gait + '_torque' +'.png', bbox_inches='tight')

    plt.show()





minicheetah_walking_trot = partial(MiniCheetah, gait="walking_trot")
minicheetah_running_trot = partial(MiniCheetah, gait="running_trot")
minicheetah_rotary_gallop = partial(MiniCheetah, gait="rotary_gallop")
minicheetah_bound = partial(MiniCheetah, gait="bound")    

# plot_torque(minicheetah_walking_trot)
# plot_torque(minicheetah_running_trot)
# plot_torque(minicheetah_rotary_gallop)
plot_torque(minicheetah_bound)

import time
while True:
    time.sleep(2)

