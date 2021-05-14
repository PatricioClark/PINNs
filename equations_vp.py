import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def NS3D(model, coords, params, separate_terms=False):
    """ NS 3D equations """

    PX    = params[0]
    nu    = params[1]

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            with tf.GradientTape(persistent=True) as tape0:
                tape0.watch(coords)
                Yp = model(coords)[0]
                f1  = Yp[:,0] 
                f2  = Yp[:,1] 
                f3  = Yp[:,2] 
                p   = Yp[:,3]

            # Zeroth derivatives
            grad_f1 = tape0.gradient(f1, coords)
            f1_t = grad_f1[:,0]
            f1_x = grad_f1[:,1]
            f1_y = grad_f1[:,2]
            f1_z = grad_f1[:,3]

            grad_f2 = tape0.gradient(f2, coords)
            f2_t = grad_f2[:,0]
            f2_x = grad_f2[:,1]
            f2_y = grad_f2[:,2]
            f2_z = grad_f2[:,3]

            grad_f3 = tape0.gradient(f3, coords)
            f3_t = grad_f3[:,0]
            f3_x = grad_f3[:,1]
            f3_y = grad_f3[:,2]
            f3_z = grad_f3[:,3]

            del tape0

            # Calculte curl
            u = (f3_y - f2_z)
            v = (f1_z - f3_x)
            w = (f2_x - f1_y)
            p = p             

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_t = grad_u[:,0]
        u_x = grad_u[:,1]
        u_y = grad_u[:,2]
        u_z = grad_u[:,3]

        grad_v = tape1.gradient(v, coords)
        v_t = grad_v[:,0]
        v_x = grad_v[:,1]
        v_y = grad_v[:,2]
        v_z = grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_t = grad_w[:,0]
        w_x = grad_w[:,1]
        w_y = grad_w[:,2]
        w_z = grad_w[:,3]

        grad_p = tape1.gradient(p, coords)
        p_x = grad_p[:,1]
        p_y = grad_p[:,2]
        p_z = grad_p[:,3]
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

    # Equations to be enforced
    if not separate_terms:
        # f0 = u_x + v_y + w_z
        f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz))
        f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz))
        f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz))
            
        # return [f0, f1, f2, f3]
        return [f1, f2, f3]
    else:
        return ([u_x, v_y, w_z],
                [u_t,
                 u*u_x, v*u_y, w*u_z,
                 p_x, PX*tf.ones(p_x.shape, dtype=p_x.dtype),
                -nu*u_xx, -nu*u_yy, -nu*u_zz],
                [v_t,
                 u*v_x, v*v_y, w*v_z,
                 p_y, 0*tf.ones(p_y.shape, dtype=p_y.dtype),
                -nu*v_xx, -nu*v_yy, -nu*v_zz],
                [w_t,
                 u*w_x, v*w_y, w*w_z,
                 p_z, 0*tf.ones(p_z.shape, dtype=p_z.dtype),
                -nu*w_xx, -nu*w_yy, -nu*w_zz],
                )
