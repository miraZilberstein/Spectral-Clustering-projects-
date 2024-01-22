from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    iris = datasets.load_iris()  # Load Iris dataset
    x = range(1, 11)  # Cluster range
    y = []

    # Compute inertia for each k
    for i in x:
        model = KMeans(n_clusters=i, init='k-means++', random_state=0)
        model.fit(iris.data)
        y.append(model.inertia_)

    # Plotting
    plt.plot(x, y)
    plt.title('Elbow Method for optimal "K"')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.xticks(x)
    # Elbow point annotation
    plt.annotate('Elbow Point', xy=(3, y[2]), xytext=(3, y[2]+150), arrowprops=dict(facecolor='green', shrink=0.05))
    plt.savefig('elbow.png')  # Save plot
    plt.close()  # Close plot

if __name__ == "__main__":
    main()
