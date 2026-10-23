import tensorflow as tf

@tf.function
def RB2D(model, coords, eq_params):
    """ 2D Rayleigh-Benard (Oberbeck-Boussinesq) equations

    Free-fall units: lengths in units of the cell height, velocities in units
    of the free-fall velocity and temperature in units of the temperature
    difference between the plates. Gravity points along -z.

    coords = (t, x, z), model outputs = (u, w, T, p), eq_params = [Ra, Pr].
    """

    Ra = eq_params[0]
    Pr = eq_params[1]
    nu    = tf.sqrt(Pr/Ra)
    kappa = 1.0/tf.sqrt(Ra*Pr)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            yp = model(coords)[0]
            u  = yp[:,0]
            w  = yp[:,1]
            T  = yp[:,2]
            p  = yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_t = grad_u[:,0]
        u_x = grad_u[:,1]
        u_z = grad_u[:,2]

        grad_w = tape1.gradient(w, coords)
        w_t = grad_w[:,0]
        w_x = grad_w[:,1]
        w_z = grad_w[:,2]

        grad_T = tape1.gradient(T, coords)
        T_t = grad_T[:,0]
        T_x = grad_T[:,1]
        T_z = grad_T[:,2]

        grad_p = tape1.gradient(p, coords)
        p_x = grad_p[:,1]
        p_z = grad_p[:,2]
        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, coords)[:,1]
    w_xx = tape2.gradient(w_x, coords)[:,1]
    T_xx = tape2.gradient(T_x, coords)[:,1]

    u_zz = tape2.gradient(u_z, coords)[:,2]
    w_zz = tape2.gradient(w_z, coords)[:,2]
    T_zz = tape2.gradient(T_z, coords)[:,2]
    del tape2

    # Equations to be enforced
    f0 = u_x + w_z
    f1 = u_t + u*u_x + w*u_z + p_x - nu*(u_xx+u_zz)
    f2 = w_t + u*w_x + w*w_z + p_z - nu*(w_xx+w_zz) - T
    f3 = T_t + u*T_x + w*T_z - kappa*(T_xx+T_zz)

    return [f0, f1, f2, f3]
