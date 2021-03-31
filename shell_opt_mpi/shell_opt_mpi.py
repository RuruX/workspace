from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

from set_fea import *
from states_comp import StatesComp
from objective_comp import ObjectiveComp
from constraint_comp import ConstraintComp
import cProfile, pstats, io

def profile(filename=None, comm=MPI.comm_world):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.Get_rank())
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator
    

class ShellGroup(om.Group):

    def initialize(self):
        self.options.declare('fea')

    def setup(self):
        self.fea = fea = self.options['fea']
        
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('h', shape=fea.dof_f)
        self.add_subsystem('inputs_comp', inputs_comp)

        comp_1 = StatesComp(fea=fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=fea)
        self.add_subsystem('objective_comp', comp_2)

        comp_3 = ConstraintComp(fea=fea)
        self.add_subsystem('constraint_comp', comp_3)

        self.connect('inputs_comp.h','states_comp.h')
        self.connect('inputs_comp.h','constraint_comp.h')
        self.connect('states_comp.displacements',
                     'objective_comp.displacements')

        self.add_design_var('inputs_comp.h', lower=1e-3, upper=0.5)
        self.add_objective('objective_comp.objective')
        
        # width = 2, length = 20, average_h = 0.2
        self.add_constraint('constraint_comp.constraint', equals=8.)



if __name__ == '__main__':

    
    mesh = Mesh()
    filename = "plate3.xdmf"
    file = XDMFFile(mesh.mpi_comm(),filename)
    file.read(mesh)
    
    fea = set_fea(mesh)
    group = ShellGroup(fea=fea)

    prob = om.Problem(model=group)
    rank = MPI.comm_world.Get_rank()
    num_pc = MPI.comm_world.Get_size()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-10
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-9
    prob.setup()
    prob.run_model()
    print(prob['objective_comp.objective'],'\n')
#    prob.check_totals(compact_print=True)

    @profile(filename="profile_out")
    def main(prob):

        import timeit
        start = timeit.default_timer()
        prob.run_driver()
        stop = timeit.default_timer()
        compliance = prob['objective_comp.objective']
        volume = prob['constraint_comp.constraint']
        print('compliance:', compliance)
        print('volume:', volume)
#        if rank == 0:
#            print("Program finished!")
#            File = open('check_results.txt', 'a')
#            print(type(num_pc))
#            outputs = '\n{1:2d}     {2:.3f}     {3:f}'.format(
#                        num_pc, stop-start, compliance)
#            File.write(outputs)
#            File.close()

    cProfile.run('main(prob)', "profile_out")


    u_mid,theta = fea.w.split(True)
    u_mid.rename("u","u")
    theta.rename("t","t")
    File('h.pvd') << fea.h
    File('t.pvd') << theta
    File('u.pvd') << u_mid


