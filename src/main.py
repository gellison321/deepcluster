from centroid_agent_game import Agent, Environment, train
from utils import blobs, plt

def main():

    k = 4

    X_train, X_test, Y_train, Y_test = blobs(n_samples=500, cluster_std=.5,
                                             n_features=2, centers=k, test_size=0.2, random_state=100)

    env = Environment(X_train, k)
    agent = Agent(X_train, k)
    train(env, agent, episodes=500,epochs=100)

    C = agent.select_action(X_train.flatten())

    for x in X_train:
        plt.scatter(x[0].item(), x[1].item(), c='b')
    for c in C:
        plt.scatter(c[0].item(), c[1].item(), c='r')
    plt.show(block=True)


if __name__ == '__main__':
    main()