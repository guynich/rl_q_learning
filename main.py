import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

SAVE_ANIMATION = False  # Set to True to save the animation as a GIF
                        # (no plot will be shown, saving takes time).

# Environment and Q-learning settings
grid_size = 4
n_actions = 4
episodes = 100
max_steps = 50
learning_rate = 0.1
discount = 0.95
epsilon = 0.1
goal = (3, 3)

# Action definitions: up, down, left, right (dy, dx)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

q_table = np.zeros((grid_size, grid_size, n_actions))
agent_pos = [0, 0]
episode = 0
step = 0

# Set up plot
initial_title = "Learned policy (Q-table)"
fig, (ax_env, ax_policy) = plt.subplots(1, 2, figsize=(10, 5))

# Environment plot setup
env_grid = np.zeros((grid_size, grid_size))
env_grid[goal] = 0.5  # mark goal as light blue in grayscale
env_plot = ax_env.imshow(env_grid, cmap="Blues", vmin=0, vmax=1)
(agent_marker,) = ax_env.plot([], [], "ro", markersize=10)

ax_env.set_title("Reinforcement learning: agent movement")
ax_env.set_xticks(np.arange(grid_size))
ax_env.set_yticks(np.arange(grid_size))
ax_env.set_xticklabels(np.arange(grid_size))
ax_env.set_yticklabels(np.arange(grid_size))
ax_env.grid(True)

# Policy arrow plot setup
X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
U = np.zeros_like(X, dtype=float)
V = np.zeros_like(Y, dtype=float)
quiver = ax_policy.quiver(
    X, Y, U, V, angles="xy", scale_units="xy", scale=1, pivot="middle"
)
ax_policy.set_title(initial_title)
ax_policy.set_xticks(np.arange(grid_size))
ax_policy.set_yticks(np.arange(grid_size))
ax_policy.set_xticklabels(np.arange(grid_size))
ax_policy.set_yticklabels(np.arange(grid_size))
ax_policy.grid(True)
ax_policy.invert_yaxis()


def reset():
    global agent_pos, step
    agent_pos = [0, 0]
    step = 0


def choose_action(pos):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q_table[pos[0], pos[1]])


def step_env(pos, action):
    dy, dx = actions[action]
    ny = min(max(pos[0] + dy, 0), grid_size - 1)
    nx = min(max(pos[1] + dx, 0), grid_size - 1)
    done = (ny, nx) == goal
    reward = 10 if done else -1
    return [ny, nx], reward, done


def update(frame):
    global agent_pos, episode, step, U, V

    if episode >= episodes:
        return

    y, x = agent_pos
    action = choose_action(agent_pos)
    new_pos, reward, done = step_env(agent_pos, action)

    ny, nx = new_pos
    best_future = np.max(q_table[ny, nx])
    q_table[y, x, action] += learning_rate * (
        reward + discount * best_future - q_table[y, x, action]
    )

    agent_pos = new_pos
    step += 1
    agent_marker.set_data([agent_pos[1]], [agent_pos[0]])

    if done or step >= max_steps:
        episode += 1
        reset()

        # Update policy arrows
        for i in range(grid_size):
            for j in range(grid_size):
                best_a = np.argmax(q_table[i, j])
                dy, dx = actions[best_a]
                U[i, j] = dx
                V[i, j] = dy  # No inversion now — y-axis is already flipped visually
        quiver.set_UVC(U, V)
        ax_policy.set_title(f"{initial_title} after {episode} episodes")


ani = animation.FuncAnimation(
    fig, update, frames=episodes * max_steps, interval=50, repeat=False
)
plt.tight_layout()
if SAVE_ANIMATION:
    ani.save("q_learning_animation.gif",
             writer='pillow',
             fps=10,
             dpi=100,
             savefig_kwargs={'facecolor': 'white'})
    plt.close()
else:
    plt.show()
