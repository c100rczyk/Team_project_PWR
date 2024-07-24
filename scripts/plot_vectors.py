import matplotlib.pyplot as plt
import numpy as np


def euclidean_distance(vects):
    """
    Znalezienie odległości euklidesowej pomiędzy dwoma wektorami:

    Arguments:
        vects: Lista zawierająca dwa wektory

    Returns:
        Odległość euklidesowa pomiędzy podanymi wektorami
    """
    x, y = vects
    return np.sqrt(np.sum(np.square(x - y)))


def find_nearest_vector(target_vector, vector_list):
    """
    Funkcja znajdująca wektor z listy wektorów, który jest najbliższy do danego wektora.

    Arguments:
        target_vector: Wektor, do którego szukamy najbliższego wektora
        vector_list: Lista zawierająca wektory, wśród których szukamy najbliższego

    Returns:
        Najbliższy wektor z listy
    """
    min_distance = float("inf")
    nearest_vector = None

    for vector in vector_list:
        distance = euclidean_distance((target_vector, vector))
        if distance < min_distance:
            min_distance = distance
            nearest_vector = vector

    return nearest_vector


def plot_vectors(target_vector, vector_list, nearest_vector):
    """
    Funkcja wyświetlająca wektory na wykresie w trójwymiarowej przestrzeni.

    Arguments:
        target_vector: Wektor, który będzie oznaczony jako punkt startowy (czerwony)
        vector_list: Lista zawierająca wektory, które zostaną wyświetlone na wykresie
        nearest_vector: Wektor, który zostanie oznaczony jako najbliższy do target_vector (opcjonalny, zielony)

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Wyświetlanie wektorów
    for vector in vector_list:
        ax.quiver(
            0,
            0,
            0,
            vector[0],
            vector[1],
            vector[2],
            color="blue",
            arrow_length_ratio=0.1,
        )

    # Wyświetlanie wektora docelowego (target_vector)
    ax.quiver(
        0,
        0,
        0,
        target_vector[0],
        target_vector[1],
        target_vector[2],
        color="red",
        arrow_length_ratio=0.1,
    )

    # Wyświetlanie najbliższego wektora (jeśli został podany)
    if nearest_vector is not None:
        ax.quiver(
            0,
            0,
            0,
            nearest_vector[0],
            nearest_vector[1],
            nearest_vector[2],
            color="green",
            arrow_length_ratio=0.1,
        )

    # Ustawienia wykresu
    ax.set_xlim(
        [
            -max(max(target_vector), max(v[0] for v in vector_list)),
            max(max(target_vector), max(v[0] for v in vector_list)),
        ]
    )
    ax.set_ylim(
        [
            -max(max(target_vector), max(v[1] for v in vector_list)),
            max(max(target_vector), max(v[1] for v in vector_list)),
        ]
    )
    ax.set_zlim(
        [
            -max(max(target_vector), max(v[2] for v in vector_list)),
            max(max(target_vector), max(v[2] for v in vector_list)),
        ]
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=30)  # Ustawienie widoku

    plt.show()


# Przykładowe użycie
target_vector = np.array([1, 2, 3])
vector_list = [np.array([2, 2, 3]), np.array([-3, -2, -1]), np.array([10, 20, 30])]
nearest_vector = find_nearest_vector(target_vector, vector_list)
plot_vectors(target_vector, vector_list, nearest_vector)
print("Najbliższy wektor:", nearest_vector)
