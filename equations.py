import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import numpy as np

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
          tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau31_x + tau32_y + tau33_z)

    return [f0, f1, f2, f3]

@tf.function
def LES3DSmagMason(model, coords, params):
    """ LES 3D equations with Smagorinsky model and Mason wall damping"""

    PX    = params[0]
    nu    = params[1]
    delta = params[2]
    c_s   = params[3]
    kappa = params[4]
    n_mwd = params[5]
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

        l0 = c_s*delta
        damped = (l0**(-n_mwd) + (kappa*coords[:,2])**(-n_mwd))**(-1.0/n_mwd)
        eddy_viscosity = (damped)**2*tf.sqrt(2*(S11**2+2*S12**2+2*S13**2+
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
          tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau31_x + tau32_y + tau33_z)

    return [f0, f1, f2, f3]

@tf.function
def LES3DSmagMasonNorm(model, coords, params):
    """ LES 3D equations with Smagorinsky model and Mason wall damping 
        These equations re-normalize the inputs and outputs used from the NN """

    PX    = params[0]
    nu    = params[1]
    delta = params[2]
    c_s   = params[3]
    kappa = params[4]
    n_mwd = params[5]
    U     = params[6]
    V     = params[7]
    W     = params[8]
    sig_u = params[9]
    sig_v = params[10]
    sig_w = params[11]
    t_min = params[12]
    t_max = params[13]
    x_min = params[14]
    x_max = params[15]
    y_min = params[16]
    y_max = params[17]
    z_min = params[18]
    z_max = params[19]
    sig_eq0 = params[20]
    sig_eq1 = params[21]
    sig_eq2 = params[22]
    sig_eq3 = params[23]

    jt = 0.5*(t_max-t_min)
    jx = 0.5*(x_max-x_min)
    jy = 0.5*(y_max-y_min)
    jz = 0.5*(z_max-z_min)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = sig_u*Yp[:,0] + U
            v  = sig_v*Yp[:,1] + V
            w  = sig_w*Yp[:,2] + W
            p  = Yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_x = jx*grad_u[:,1]
        u_y = jy*grad_u[:,2]
        u_z = jz*grad_u[:,3]

        grad_v = jy*tape1.gradient(v, coords)
        v_x = jx*grad_v[:,1]
        v_y = jy*grad_v[:,2]
        v_z = jz*grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_x = jx*grad_w[:,1]
        w_y = jy*grad_w[:,2]
        w_z = jz*grad_w[:,3]

        grad_p = tape1.gradient(p, coords)

        S11 = u_x
        S12 = 0.5*(u_y+v_x)
        S13 = 0.5*(u_z+w_x)
        S22 = v_y
        S23 = 0.5*(v_z+w_y)
        S33 = w_z

        l0 = c_s*delta
        damped = (l0**(-n_mwd) + (kappa*coords[:,2])**(-n_mwd))**(-1.0/n_mwd)
        eddy_viscosity = (damped)**2*tf.sqrt(2*(S11**2+2*S12**2+2*S13**2+
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
    u_xx = jx*tape2.gradient(u_x, coords)[:,1]
    v_xx = jx*tape2.gradient(v_x, coords)[:,1]
    w_xx = jx*tape2.gradient(w_x, coords)[:,1]

    u_yy = jy*tape2.gradient(u_y, coords)[:,2]
    v_yy = jy*tape2.gradient(v_y, coords)[:,2]
    w_yy = jy*tape2.gradient(w_y, coords)[:,2]

    u_zz = jz*tape2.gradient(u_z, coords)[:,3]
    v_zz = jz*tape2.gradient(v_z, coords)[:,3]
    w_zz = jz*tape2.gradient(w_z, coords)[:,3]
    
    tau11_x = jx*tape2.gradient(tau11, coords)[:,1]
    tau21_x = jx*tape2.gradient(tau12, coords)[:,1]
    tau31_x = jx*tape2.gradient(tau13, coords)[:,1]

    tau12_y = jy*tape2.gradient(tau12, coords)[:,2]
    tau22_y = jy*tape2.gradient(tau22, coords)[:,2]
    tau32_y = jy*tape2.gradient(tau23, coords)[:,2]

    tau13_z = jz*tape2.gradient(tau13, coords)[:,3]
    tau23_z = jz*tape2.gradient(tau23, coords)[:,3]
    tau33_z = jz*tape2.gradient(tau33, coords)[:,3]
    del tape2

    # First derivates that are not differentiated a second time
    u_t = jt*grad_u[:,0]
    v_t = jt*grad_v[:,0]
    w_t = jt*grad_w[:,0]

    p_x = jx*grad_p[:,1]
    p_y = jy*grad_p[:,2]
    p_z = jz*grad_p[:,3]

    # Equations to be enforced
    f0 = u_x + v_y + w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz) +
          tau11_x + tau12_y + tau13_z)
    f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz) +
          tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau31_x + tau32_y + tau33_z)

    return [f0/sig_eq0, f1/sig_eq1, f2/sig_eq2, f3/sig_eq3]

@tf.function
def LES3DSmagMasonNormTerms(model, coords, params):
    """ LES 3D equations with Smagorinsky model and Mason wall damping 
        These equations re-normalize the inputs and outputs used from the NN 
        Each term is outputted independently now """

    PX    = params[0]
    nu    = params[1]
    delta = params[2]
    c_s   = params[3]
    kappa = params[4]
    n_mwd = params[5]
    U     = params[6]
    V     = params[7]
    W     = params[8]
    sig_u = params[9]
    sig_v = params[10]
    sig_w = params[11]
    t_min = params[12]
    t_max = params[13]
    x_min = params[14]
    x_max = params[15]
    y_min = params[16]
    y_max = params[17]
    z_min = params[18]
    z_max = params[19]
    sig_eq0 = params[20]
    sig_eq1 = params[21]
    sig_eq2 = params[22]
    sig_eq3 = params[23]

    jt = 0.5*(t_max-t_min)
    jx = 0.5*(x_max-x_min)
    jy = 0.5*(y_max-y_min)
    jz = 0.5*(z_max-z_min)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = sig_u*Yp[:,0] + U
            v  = sig_v*Yp[:,1] + V
            w  = sig_w*Yp[:,2] + W
            p  = Yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_x = jx*grad_u[:,1]
        u_y = jy*grad_u[:,2]
        u_z = jz*grad_u[:,3]

        grad_v = jy*tape1.gradient(v, coords)
        v_x = jx*grad_v[:,1]
        v_y = jy*grad_v[:,2]
        v_z = jz*grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_x = jx*grad_w[:,1]
        w_y = jy*grad_w[:,2]
        w_z = jz*grad_w[:,3]

        grad_p = tape1.gradient(p, coords)

        S11 = u_x
        S12 = 0.5*(u_y+v_x)
        S13 = 0.5*(u_z+w_x)
        S22 = v_y
        S23 = 0.5*(v_z+w_y)
        S33 = w_z

        l0 = c_s*delta
        damped = (l0**(-n_mwd) + (kappa*coords[:,2])**(-n_mwd))**(-1.0/n_mwd)
        eddy_viscosity = (damped)**2*tf.sqrt(2*(S11**2+2*S12**2+2*S13**2+
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
    u_xx = jx*tape2.gradient(u_x, coords)[:,1]
    v_xx = jx*tape2.gradient(v_x, coords)[:,1]
    w_xx = jx*tape2.gradient(w_x, coords)[:,1]

    u_yy = jy*tape2.gradient(u_y, coords)[:,2]
    v_yy = jy*tape2.gradient(v_y, coords)[:,2]
    w_yy = jy*tape2.gradient(w_y, coords)[:,2]

    u_zz = jz*tape2.gradient(u_z, coords)[:,3]
    v_zz = jz*tape2.gradient(v_z, coords)[:,3]
    w_zz = jz*tape2.gradient(w_z, coords)[:,3]
    
    tau11_x = jx*tape2.gradient(tau11, coords)[:,1]
    tau21_x = jx*tape2.gradient(tau12, coords)[:,1]
    tau31_x = jx*tape2.gradient(tau13, coords)[:,1]

    tau12_y = jy*tape2.gradient(tau12, coords)[:,2]
    tau22_y = jy*tape2.gradient(tau22, coords)[:,2]
    tau32_y = jy*tape2.gradient(tau23, coords)[:,2]

    tau13_z = jz*tape2.gradient(tau13, coords)[:,3]
    tau23_z = jz*tape2.gradient(tau23, coords)[:,3]
    tau33_z = jz*tape2.gradient(tau33, coords)[:,3]
    del tape2

    # First derivates that are not differentiated a second time
    u_t = jt*grad_u[:,0]
    v_t = jt*grad_v[:,0]
    w_t = jt*grad_w[:,0]

    p_x = jx*grad_p[:,1]
    p_y = jy*grad_p[:,2]
    p_z = jz*grad_p[:,3]

    # Equations to be enforced
    f0 = u_x + v_y + w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz) +
          tau11_x + tau12_y + tau13_z)
    f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz) +
          tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau31_x + tau32_y + tau33_z)

    return ([u_x, v_y, w_z],
            [u_t, u*u_x, v*u_y, w*u_z, p_x, PX,
                -nu*u_xx, -nu*u_yy, -nu*u_zz,
                 tau11_x,  tau12_y,  tau13_z],
            [v_t, u*v_x, v*v_y, v*u_z, p_y, 0
                -nu*v_xx, -nu*v_yy, -nu*v_zz,
                 tau21_x,  tau22_y,  tau23_z],
            [w_t, u*w_x, v*w_y, w*w_z, p_z, 0,
                -nu*w_xx, -nu*w_yy, -nu*w_zz,
                 tau31_x,  tau32_y,  tau33_z])

@tf.function
def LES3DNonl(model, coords, params):
    """ LES 3D equations with Nonl model"""

    PX     = params[0]
    nu     = params[1]
    delta  = params[2]
    c_s    = params[3]
    points = params[4]
    r_int  = params[5]
    pre    = params[6]
    # pre = dr**(1.0-alpha)/(4.0*np.pi*tf.math.exp(tf.lgamma(2.0-alpha)))

    Nint = len(r_int)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = Yp[:,0]
            v  = Yp[:,1]
            w  = Yp[:,2]
            p  = Yp[:,3]

            disps = [model(coords+points[ii])[0] for ii in range(Nint)]
            ud    = [dd[:,0] for dd in disps]
            vd    = [dd[:,1] for dd in disps]
            wd    = [dd[:,2] for dd in disps]

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

        grad_ud = [tape1.gradient(dd, coords) for dd in ud]
        ud_x    = [gg[:,1] for gg in grad_ud]
        ud_y    = [gg[:,2] for gg in grad_ud]
        ud_z    = [gg[:,3] for gg in grad_ud]
        
        grad_vd = [tape1.gradient(dd, coords) for dd in vd]
        vd_x    = [gg[:,1] for gg in grad_vd]
        vd_y    = [gg[:,2] for gg in grad_vd]
        vd_z    = [gg[:,3] for gg in grad_vd]

        grad_wd = [tape1.gradient(dd, coords) for dd in wd]
        wd_x    = [gg[:,1] for gg in grad_wd]
        wd_y    = [gg[:,2] for gg in grad_wd]
        wd_z    = [gg[:,3] for gg in grad_wd]

        Sd11 = ud_x
        Sd12 = 0.5*(ud_y+vd_x)
        Sd13 = 0.5*(ud_z+wd_x)
        Sd22 = vd_y
        Sd23 = 0.5*(vd_z+wd_y)
        Sd33 = wd_z

        Salpha11 = pre*tf.add_n([r_int[i]*ud_x[i] for i in range(Nint)])
        Salpha12 = pre*tf.add_n([r_int[i]*0.5*(ud_y[i]+vd_x[i]) for i in range(Nint)])
        Salpha13 = pre*tf.add_n([r_int[i]*0.5*(ud_z[i]+wd_x[i]) for i in range(Nint)])
        Salpha22 = pre*tf.add_n([r_int[i]*vd_y[i] for i in range(Nint)])
        Salpha23 = pre*tf.add_n([r_int[i]*0.5*(vd_z[i]+wd_y[i]) for i in range(Nint)])
        Salpha33 = pre*tf.add_n([r_int[i]*wd_z[i] for i in range(Nint)])

        # uso la eddy viscosity local
        tau11 = -2*eddy_viscosity*Salpha11
        tau12 = -2*eddy_viscosity*Salpha12
        tau13 = -2*eddy_viscosity*Salpha13
        tau22 = -2*eddy_viscosity*Salpha22
        tau23 = -2*eddy_viscosity*Salpha23
        tau33 = -2*eddy_viscosity*Salpha33

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
          tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz) +
          tau31_x + tau32_y + tau33_z)

    return [f0, f1, f2, f3]
