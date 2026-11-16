
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

tf.keras.backend.set_floatx('float32')

R = 3700.0
g = 21.85
lapse_rate = 0.0065

P0 = 101325.0
T0 = 288.0

xmin = 0.0
xmax = 5000.0
lb = tf.constant(xmin, dtype='float32')
ub = tf.constant(xmax, dtype='float32')

x_bc = tf.zeros((1, 1), dtype='float32')

np.random.seed(42)
n_collocation = 2000
z_collocation = np.random.uniform(
    0, 5000, (n_collocation, 1)
).astype(np.float32)

x_r = tf.constant(z_collocation)


def mymodel(num_hidden_layer=4, num_neurons_per_layer=64):

    model = keras.Sequential()

    scaling_layer = keras.layers.Lambda(
        lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0,
        input_shape=(1,)
    )

    model.add(scaling_layer)

    for i in range(num_hidden_layer):
        model.add(
            keras.layers.Dense(
                num_neurons_per_layer,
                activation='tanh',
                kernel_initializer='glorot_normal'
            )
        )

    model.add(keras.layers.Dense(2, activation='linear'))

    def transform_output(x):

        log_P_normalized, T_normalized = tf.split(
            x, 2, axis=1
        )

        P = P0 * tf.exp(log_P_normalized)
        T = T0 * tf.nn.sigmoid(T_normalized) * 2.0

        return tf.concat([P, T], axis=1)

    model.add(keras.layers.Lambda(transform_output))

    return model


def compute_physics_residuals(model, x_r):

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x_r)

        outputs = model(x_r)
        P, T = tf.split(outputs, 2, axis=1)

    dP_dz = tape.gradient(P, x_r)
    dT_dz = tape.gradient(T, x_r)

    del tape

    barometric_residual = dP_dz + (P * g) / (R * T)
    lapse_residual = dT_dz + lapse_rate

    return barometric_residual, lapse_residual, P, T


def compute_loss(
    model,
    x_r,
    x_bc,
    lambda_lapse=100.0,
    lambda_bc=1000.0
):

    baro_res, lapse_res, P_interior, T_interior = (
        compute_physics_residuals(model, x_r)
    )

    loss_barometric = tf.reduce_mean(tf.square(baro_res))
    loss_lapse = tf.reduce_mean(tf.square(lapse_res))

    outputs_bc = model(x_bc)
    P_bc, T_bc = tf.split(outputs_bc, 2, axis=1)

    bc_loss_P = tf.reduce_mean(tf.square(P_bc - P0))
    bc_loss_T = tf.reduce_mean(tf.square(T_bc - T0))

    bc_loss = bc_loss_P + bc_loss_T

    loss_total = (
        loss_barometric
        + lambda_lapse * loss_lapse
        + lambda_bc * bc_loss
    )

    return loss_total, loss_barometric, loss_lapse, bc_loss


model = mymodel(
    num_hidden_layer=4,
    num_neurons_per_layer=64
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule
)


@tf.function
def train_step(x_r, x_bc):

    with tf.GradientTape() as tape:

        loss, _, _, _ = compute_loss(
            model, x_r, x_bc
        )

    gradients = tape.gradient(
        loss, model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    return loss


N_epochs = 30000

loss_history = {
    'total': [],
    'barometric': [],
    'lapse': [],
    'bc': []
}


print("=" * 70)
print("PINN Training - Barometric Equation")
print("=" * 70)
print(f"Number of collocation points: {n_collocation}")
print(f"Number of epochs: {N_epochs}")
print("Architecture: 4 hidden layers x 64 neurons")
print("=" * 70 + "\n")


for epoch in range(N_epochs + 1):

    loss = train_step(x_r, x_bc)

    if epoch % 1000 == 0:

        total_loss, baro_loss, lapse_loss, bc_loss = (
            compute_loss(model, x_r, x_bc)
        )

        loss_history['total'].append(total_loss.numpy())
        loss_history['barometric'].append(baro_loss.numpy())
        loss_history['lapse'].append(lapse_loss.numpy())
        loss_history['bc'].append(bc_loss.numpy())

        print(
            f'Epoch {epoch:5d} | '
            f'Total: {total_loss.numpy():.3e} | '
            f'Barometric: {baro_loss.numpy():.3e} | '
            f'Lapse Rate: {lapse_loss.numpy():.3e} | '
            f'Boundary: {bc_loss.numpy():.3e}'
        )


print("\n" + "=" * 70)
print("Training completed successfully!")
print("=" * 70 + "\n")


z_test = np.linspace(
    0, 5000, 500
).reshape(-1, 1).astype(np.float32)

outputs_pred = model(z_test).numpy()

P_pred = outputs_pred[:, 0:1]
T_pred = outputs_pred[:, 1:2]

T_analytical = T0 - lapse_rate * z_test

P_analytical = P0 * (
    T_analytical / T0
) ** (g / (R * lapse_rate))

rho_pred = P_pred / (R * T_pred)
rho_analytical = P_analytical / (R * T_analytical)

error_P = (
    np.abs(P_pred - P_analytical)
    / P_analytical * 100
)

error_T = (
    np.abs(T_pred - T_analytical)
    / T_analytical * 100
)


print(
    f"Mean Pressure Error: {np.mean(error_P):.2f}%"
)

print(
    f"Mean Temperature Error: {np.mean(error_T):.2f}%"
)

print(
    f"Maximum Pressure Error: {np.max(error_P):.2f}%"
)

print(
    f"Maximum Temperature Error: {np.max(error_T):.2f}%\n"
)


fig = plt.figure(figsize=(18, 10))


ax1 = plt.subplot(2, 4, 1)

plt.plot(
    P_pred / 1000,
    z_test,
    'b-',
    linewidth=2,
    label="PINN"
)

plt.plot(
    P_analytical / 1000,
    z_test,
    'r--',
    linewidth=2,
    label="Analytical"
)

plt.xlabel("Pressure (kPa)", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Pressure Profile", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


ax2 = plt.subplot(2, 4, 2)

plt.plot(
    T_pred,
    z_test,
    'g-',
    linewidth=2,
    label="PINN"
)

plt.plot(
    T_analytical,
    z_test,
    'r--',
    linewidth=2,
    label="Analytical"
)

plt.xlabel("Temperature (K)", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Temperature Profile", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


ax3 = plt.subplot(2, 4, 3)

plt.plot(
    rho_pred,
    z_test,
    'm-',
    linewidth=2,
    label="PINN"
)

plt.plot(
    rho_analytical,
    z_test,
    'r--',
    linewidth=2,
    label="Analytical"
)

plt.xlabel("Density (kg/m³)", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Density Profile", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


ax4 = plt.subplot(2, 4, 4)

plt.plot(
    error_P,
    z_test,
    'b-',
    linewidth=2
)

plt.xlabel("Relative Error (%)", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Pressure Error", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)


ax5 = plt.subplot(2, 4, 5)

epochs_plot = np.arange(
    0, N_epochs + 1, 1000
)

plt.semilogy(
    epochs_plot,
    loss_history['total'],
    'b-',
    linewidth=2,
    label='Total'
)

plt.semilogy(
    epochs_plot,
    loss_history['barometric'],
    'r--',
    linewidth=1.5,
    label='Barometric'
)

plt.semilogy(
    epochs_plot,
    loss_history['lapse'],
    'g--',
    linewidth=1.5,
    label='Lapse Rate'
)

plt.semilogy(
    epochs_plot,
    loss_history['bc'],
    'm--',
    linewidth=1.5,
    label='Boundary Condition'
)

plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.title("Training History", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


ax6 = plt.subplot(2, 4, 6)

plt.plot(
    error_T,
    z_test,
    'g-',
    linewidth=2
)

plt.xlabel("Relative Error (%)", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Temperature Error", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)


ax7 = plt.subplot(2, 4, 7)

plt.plot(
    P_pred / P0,
    z_test,
    'b-',
    linewidth=2,
    label="PINN"
)

plt.plot(
    P_analytical / P0,
    z_test,
    'r--',
    linewidth=2,
    label="Analytical"
)

plt.xlabel("P/P0", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Normalized Pressure", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


ax8 = plt.subplot(2, 4, 8)

plt.plot(
    T_pred / T0,
    z_test,
    'g-',
    linewidth=2,
    label="PINN"
)

plt.plot(
    T_analytical / T0,
    z_test,
    'r--',
    linewidth=2,
    label="Analytical"
)

plt.xlabel("T/T0", fontsize=11)
plt.ylabel("Altitude (m)", fontsize=11)
plt.title("Normalized Temperature", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)


plt.suptitle(
    "PINN Solution: Barometric Equation - Exoplanet Atmosphere",
    fontsize=14,
    fontweight='bold',
    y=0.995
)

plt.tight_layout()
plt.show()


print("=" * 70)
print("Model Performance Statistics")
print("=" * 70)

print(
    f"Surface Pressure (z=0): "
    f"{P_pred[0,0]/1000:.2f} kPa "
    f"(Analytical: {P0/1000:.2f} kPa)"
)

print(
    f"Surface Temperature: "
    f"{T_pred[0,0]:.2f} K "
    f"(Analytical: {T0:.2f} K)"
)

print(
    f"Pressure at 5000 m: "
    f"{P_pred[-1,0]/1000:.2f} kPa "
    f"(Analytical: {P_analytical[-1,0]/1000:.2f} kPa)"
)

print(
    f"Temperature at 5000 m: "
    f"{T_pred[-1,0]:.2f} K "
    f"(Analytical: {T_analytical[-1,0]:.2f} K)"
)

print("=" * 70)
