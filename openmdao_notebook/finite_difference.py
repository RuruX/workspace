def eval_res(self, edge_deltas, uhat):
    self.fea.edge_deltas = edge_deltas
    for i in range(self.input_size):
        self.fea.uhat_old.vector()[self.fea.edge_indices[i]] = edge_deltas[i]
    self.bcs = DirichletBC(self.fea.VHAT, self.fea.uhat_old, self.fea.boundaries_mf, 1000)
    update(self.fea.uhat, uhat)
    resM = assemble(self.fea.resM())
    uhat_0 = Function(self.fea.VHAT)
    uhat_0.assign(self.fea.uhat)
    self.bcs.apply(resM, uhat_0.vector())
    return resM.get_local()
    
def FD(self, inputs_0, outputs_0):
    print("="*40)
    print("CSDL: Running FD()...")
    print("="*40)
    step = 0.001
    inputs_old = inputs_0
    outputs_old = outputs_0
    residual = self.eval_res(inputs_0, outputs_0)
    dRdf = np.zeros((self.output_size, self.input_size))
    dRdu = np.zeros((self.output_size, self.output_size))
    for i in range(self.input_size):
        d_inputs_i = np.zeros(self.input_size)
        d_inputs_i[i] = step
        inputs_i = inputs_0+d_inputs_i
        residual_i = self.eval_res(inputs_i, outputs_0)
        print("*"*2,residual_i[self.fea.edge_indices[i].astype('int')])
        print(np.nonzero(residual_i-residual))
        dRdf[:,i] = (residual_i-residual)/step
    for j in range(self.output_size):    
        d_outputs_j = np.zeros(self.output_size)
        d_outputs_j[j] = step
        outputs_j = outputs_0 + d_outputs_j
        residual_j = self.eval_res(inputs_0, outputs_j)
        dRdu[:,j] = (residual_j-residual)/step
    self.dRdu = assemble(self.fea.dRm_duhat)
    self.bcs.apply(self.dRdu)
    self.M = self.fea.getBCDerivatives()
    print('Finite Difference (dR_df):', np.linalg.norm(dRdf))
    print('Compute partial (dR_df):', np.linalg.norm(self.M))
    print('Relative error (dR_df):', np.linalg.norm(dRdf-self.M)
                                        /np.linalg.norm(self.M))
    print('Finite Difference (dR_du):', np.linalg.norm(dRdu))
    print('Compute partial (dR_du):', np.linalg.norm(convertToDense(m2p(self.dRdu))))
    print('Relative error (dR_du):', np.linalg.norm(dRdu-convertToDense(m2p(self.dRdu)))
                                        /np.linalg.norm(convertToDense(m2p(self.dRdu))))
    
