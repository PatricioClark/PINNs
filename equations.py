import tensorflow as tf
tf.keras.backend.set_floatx('float64')

@tf.function
def NS3D(model, coords, params):
    """ NS 3D equations """

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = Yp[:,0]
            v  = Yp[:,1]
            w  = Yp[:,2]
            p  = Yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_x = grad_u[:,1]
        u_y = grad_u[:,2]
        u_z = grad_u[:,3]

        grad_v = tape1.gradient(v, coords)
        v_x = grad_v[:,1]
        v_y = grad_v[:,2]
        v_z = grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_x = grad_w[:,1]
        w_y = grad_w[:,2]
        w_z = grad_w[:,3]

        grad_p = tape1.gradient(p, coords)
        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, coords)[:,1]
    v_xx = tape2.gradient(v_x, coords)[:,1]
    w_xx = tape2.gradient(w_x, coords)[:,1]

    u_yy = tape2.gradient(u_y, coords)[:,2]
    v_yy = tape2.gradient(v_y, coords)[:,2]
    w_yy = tape2.gradient(w_y, coords)[:,2]

    u_zz = tape2.gradient(u_z, coords)[:,3]
    v_zz = tape2.gradient(v_z, coords)[:,3]
    w_zz = tape2.gradient(w_z, coords)[:,3]
    del tape2

    # First derivates that are not differentiated a second time
    u_t = grad_u[:,0]
    v_t = grad_v[:,0]
    w_t = grad_w[:,0]

    p_x = grad_p[:,1]
    p_y = grad_p[:,2]
    p_z = grad_p[:,3]

    # Equations to be enforced
    PX = params[0]
    nu = params[1]
    f0 = u_x + v_y + w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz))
    f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz))
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz))
        
    return [f0, f1, f2, f3]

@tf.function
def LES3DSmag(model, coords, params):
    """ LES 3D equations with Smagorinsky model"""

    PX    = params[0]
    nu    = params[1]
    delta = params[2]
    c_s   = params[3]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = Yp[:,0]
            v  = Yp[:,1]
            w  = Yp[:,2]
            p  = Yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_x = grad_u[:,1]
        u_y = grad_u[:,2]
        u_z = grad_u[:,3]

        grad_v = tape1.gradient(v, coords)
        v_x = grad_v[:,1]
        v_y = grad_v[:,2]
        v_z = grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_x = grad_w[:,1]
        w_y = grad_w[:,2]
        w_z = grad_w[:,3]

        grad_p = tape1.gradient(p, coords)

        S11 = u_x
        S12 = 0.5*(u_y+v_x)
        S13 = 0.5*(u_z+w_x)
        S22 = v_y
        S23 = 0.5*(v_z+w_y)
        S33 = w_z

        eddy_viscosity = (c_s*delta)**2*tf.sqrt(2*(S11**2+2*S12**2+2*S13**2+
                                                            S22**2+2*S23**2+
                                                                     S33**2))
        tau11 = -2*eddy_viscosity*S11
        tau12 = -2*eddy_viscosity*S12
        tau13 = -2*eddy_viscosity*S13
        tau22 = -2*eddy_viscosity*S22
        tau23 = -2*eddy_viscosity*S23
        tau33 = -2*eddy_viscosity*S33
        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, coords)[:,1]
    v_xx = tape2.gradient(v_x, coords)[:,1]
    w_xx = tape2.gradient(w_x, coords)[:,1]

    u_yy = tape2.gradient(u_y, coords)[:,2]
    v_yy = tape2.gradient(v_y, coords)[:,2]
    w_yy = tape2.gradient(w_y, coords)[:,2]

    u_zz = tape2.gradient(u_z, coords)[:,3]
    v_zz = tape2.gradient(v_z, coords)[:,3]
    w_zz = tape2.gradient(w_z, coords)[:,3]
    
    tau11_x = tape2.gradient(tau11, coords)[:,1]
    tau21_x = tape2.gradient(tau12, coords)[:,1]
    tau31_x = tape2.gradient(tau13, coords)[:,1]

    tau12_y = tape2.gradient(tau12, coords)[:,2]
    tau22_y = tape2.gradient(tau22, coords)[:,2]
    tau32_y = tape2.gradient(tau23, coords)[:,2]

    tau13_z = tape2.gradient(tau13, coords)[:,3]
    tau23_z = tape2.gradient(tau23, coords)[:,3]
    tau33_z = tape2.gradient(tau33, coords)[:,3]
    del tape2

    # First derivates that are not differentiated a second time
    u_t = grad_u[:,0]
    v_t = grad_v[:,0]
    w_t = grad_w[:,0]

    p_x = grad_p[:,1]
    p_y = grad_p[:,2]
    p_z = grad_p[:,3]

    # Equations to be enforced
    f0 = u_x+v_y+w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz) +
          tau11_x + tau12_y + tau13_z)
    f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz) +
          tau11_x + tau12_y + tau13_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau11_x + tau12_y + tau13_z)

    return [f0, f1, f2, f3]
