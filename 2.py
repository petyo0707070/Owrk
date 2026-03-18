import numpy as np
import matplotlib.pyplot as plt

class Exercise_1():
    def __init__(self):
        '''
        Since we are dealing with derivatives on function with absolute values,
        the derivaties are taken analytically as that of a normal function and multiplied by the sign of x-b
        Hence: df1(x)/dx = 1.5 * np.abs(x -b) ** 0.5 * np.sign(x - b)
        df2(x)/dx = 0.5 * np.abs(x -b) ** (- 0.5) * np.sign(x - b)
        df3(x)/dx = 3 * np.abs(x -b) ** 2 * np.sign(x - b)
        '''

        b = np.random.normal(1,1, 1) # b ~ N(1, 1)

        # Define the gradients of the 3 functions
        def grad_f1(x): # df1(x)/dx = 1.5 * np.abs(x -b) ** 0.5 * np.sign(x - b)
            return 1.5 * np.abs(x - b) ** 0.5 * np.sign(x - b)

        def grad_f2(x): # df2(x)/dx = 0.5 * np.abs(x -b) ** (- 0.5) * np.sign(x - b)
            return 0.5 * np.abs(x - b) ** (-0.5) * np.sign(x - b)

        def grad_f3(x): # df3(x)/dx = 3 * np.abs(x -b) ** 2 * np.sign(x - b)
            return 3 * np.abs(x - b) ** 2 * np.sign(x - b)

        # Get results for the gradient descend and trajectories for all 3 functions
        traj1, grad1 = self.gradient_descend(grad_f1)
        traj2, grad2 = self.gradient_descend(grad_f2)
        traj3, grad3 = self.gradient_descend(grad_f3)


        # Function f(x) = |x -b|^1.5
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(traj1, label="GD trajectory")
        plt.axhline(b, color='red', linestyle='--', label='Minimizing Value')
        plt.title("Gradient Descent on f1(x) = |x - b|^1.5")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(grad1, label = 'Gradient')
        plt.title("Gradient values for f1(x) = |x - b|^1.5")
        plt.legend()
        plt.tight_layout()
        plt.show()

        '''
        We can see that f1(x) = |x - b|^1.5 is nicely behaved the trajectory of x values is smoothed and converges within the first ~300 steps
        to its minimum value, the gradient as well decays monotonosly towards 0
        '''


        # Function f(x) = |x -b|^0.5
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(traj2, label="GD trajectory")
        plt.axhline(b, color='red', linestyle='--', label='Minimizing Value')
        plt.title("Gradient Descent on f2(x) = |x - b|^0.5")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(grad2)
        plt.title("Gradient values for f2(x) = |x - b|^0.5")
        plt.legend()
        plt.tight_layout()
        plt.show()

        '''
        In the case of f2(x) = |x - b|^0.5 the derivative -> infinity as  x -> b, which means that the gradient descend
        becomes very unstable as it approaches the glabal minimum of b, which can be seen in the constant overshooting
        and oscilation around b. Furthermore, the gradient itself is very unstable resulting in the so called "exploding gradient problem",
        which is charectarized by disproportionatly big spikes in the absolute value of the gradient leading to inability to converge
        '''

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(traj3, label="GD trajectory")
        plt.axhline(b, color='red', linestyle='--', label='Minimizing Value')
        plt.title("Gradient Descent on f3(x) = |x - b|^3")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(grad3)
        plt.title("Gradient values for f3(x) = |x - b|^3")
        plt.legend()
        plt.tight_layout()
        plt.show()

        '''
        Finally f2(x) = |x - b|^3 exhibits the 'vanishing gradient problem' i.e. the gradient becomes almost 0 the closer to b we go
        which results in very slow convergence, taking around ~ 3000 itterations. Apart from that the gradient trajectory is smooth.
        '''

    def gradient_descend(self, gradient, x_0 = 5, learning_rate = 0.01, steps = 10000):
        x = x_0 # We start at x_0 and we will continually update the value of x
        trajectory = [x] # This list will hold the trajectory of x values
        gradient_list = [np.nan] # This list will hold the gradient values

        for i in range(steps): # Perform n number of steps
            g = gradient(x)[0]

            if np.isnan(g) or np.isinf(g): # If the gradient explodes or vanishes we record it
                gradient_list.append(g)
                break

            x = x - learning_rate * g # Follow the text book formula
            trajectory.append(x)
            gradient_list.append(g)

        return trajectory, gradient_list # Return the x values and the corresponding gradient

class Exercise_2():
    def __init__(self, K = 200, batch_size = 50):

if __name__ == '__main__':
    Exercise_1()