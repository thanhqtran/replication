# extract the estimated parameters
eta, phi, theta, gamma = model1.x[:4]

# compute the implied productivity in the data
data.loc[:, 'w'] = data['y'] / (1 - phi * data['n'])

# define the functional form of the model
# implied education spending
def func_e(data, eta, phi, theta, gamma):
    w = data['w']
    e = np.where(w > theta / (eta * phi), (eta * phi * w - theta) / (1 - eta), 0.0)
    return e + theta

# implied fertility
def func_n(data, eta, phi, theta, gamma):
    w = data['w']
    n = np.where(
        w > theta / (eta * phi),
        (1 - eta) * gamma * w / ((1 + gamma) * (phi * w - theta)),
        gamma / (phi * (1 + gamma))
    )
    return n

# Sort data by w
sorted_data = data.sort_values('w')
w_sorted = sorted_data['w']
n_hat_sorted = func_n(sorted_data, eta, phi, theta, gamma)
e_hat_sorted = func_e(sorted_data, eta, phi, theta, gamma)

# plot the model-implied education and fertility against the data, 
# with w in the x-axis, 2 plots side by side, with n and e on the y-axis
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(np.log(data['w']), data['n'], label='Observed', alpha=0.5, color='orange')
plt.plot(np.log(w_sorted), n_hat_sorted, label='Model', lw=2)
plt.xlabel('log(Productivity)')
plt.ylabel('Fertility (n)')
#plt.title('Fertility vs log(Productivity)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(np.log(data['w']), np.log(data['e+theta']), label='Observed', alpha=0.5, color='orange')
plt.plot(np.log(w_sorted), np.log(e_hat_sorted), label='Model', lw=2)
plt.xlabel('log(Productivity)')
plt.ylabel('Education Spending (e+θ)')
plt.legend()

plt.tight_layout()
plt.show()
