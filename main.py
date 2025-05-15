import matplotlib.pyplot as plt
from scipy import integrate as integ
import numpy as np

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'  # Police Computer Modern
plt.rcParams['font.family'] = 'STIXGeneral'  # Autre police proche de LaTeX



# 1.1.4 Signification physique

def tracer_potentiel() -> None:
    """
    IN  >  None  
    OUT >  Affiche un graphique du potentiel harmonique réduit v_OH en fonction de x  
    """
    a = 1  # Paramètre de l'oscillateur harmonique
    x = np.linspace(0, a, 1000)  # Création de points de x entre 0 et a (1000 points)
    
    # On trace le potentiel pour plusieurs valeurs de R
    for R in [0.1, 2, 5, 10]: 
        v_OH = np.pi**2/4*R**2*(x/a-1/2)**2  # Calcul du potentiel harmonique réduit
        plt.plot(x, v_OH, label=R)  # Tracé du potentiel
        plt.grid()  # Ajout de la grille
        plt.legend(title=r"Valeurs de $R$ :")  # Légende
        plt.xlabel(r"$x$")  # Label de l'axe x
        plt.ylabel(r"Potentiel harmonique réduit $v_{OH}$")  # Label de l'axe y
        plt.title(r"Potentiel harmonique réduit dans le puits en fonction de $x$ pour différentes valeurs de $R$")
    plt.show()  # Affichage du graphique



# 1.2.1 Matrice H sous forme numérique

def hamiltonien(N: int) -> np.ndarray:
    """
    IN  >  N (int) : Taille de la matrice Hamiltonienne réduite  
    OUT >  H (numpy.ndarray) : Matrice Hamiltonienne réduite de taille (N, N)  
    """
    H = np.zeros((N, N))  # Initialisation de la matrice H de taille (N, N)
    R = 24  # Valeur du paramètre R
    for n in range(N):
        for m in range(N):
            def f(x: float) -> float: 
                """Fonction pour le calcul des éléments de la matrice H"""
                return np.sin((n+1)*np.pi*x)*np.sin((m+1)*np.pi*x)*np.pi**2/4*R**2*(x-1/2)**2 
            H[n][m] = 2 * integ.quad(f, 0, 1)[0]  # Calcul de l'intégrale de f(x) sur [0, 1]
            if n == m:
                H[n][m] += (n + 1)**2  # Ajouter les termes de la diagonale
            elif abs(H[n][m]) <= 10**(-13):  # Si la valeur est proche de zéro, on la met à zéro
                H[n][m] = 0
    return H  # Retour de la matrice H



def tracer_sinusoides() -> None:
    """
    IN  > None  
    OUT > Affiche deux graphes :  
          - Le premier avec sin(2πx'), sin(4πx') et leur produit  
          - Le second avec sin(3πx'), sin(4πx') et leur produit  
    """
    x = np.linspace(0, 1, 500)  # x' varie de 0 à 1

    n1, m = 2, 4  # n = 2, m = 4 pour le premier graphe
    n2 = 3  # n = 3 pour le deuxième graphe

    sin_n1 = np.sin(n1 * np.pi * x)  # sin(2πx')
    sin_n2 = np.sin(n2 * np.pi * x)  # sin(3πx')
    sin_m = np.sin(m * np.pi * x)  # sin(4πx')

    produit1 = sin_n1 * sin_m  # Produit des deux premières fonctions
    produit2 = sin_n2 * sin_m  # Produit des deux autres fonctions

    plt.figure(figsize=(12, 5))

    # Premier graphe : n = 2, m = 4
    plt.subplot(1, 2, 1)
    plt.plot(x, sin_n1, label=r"$\sin(2\pi x')$", linestyle='--', color='b')
    plt.plot(x, sin_m, label=r"$\sin(4\pi x')$", linestyle='-.', color='g')
    plt.plot(x, produit1, label=r"$\sin(2\pi x') \sin(4\pi x')$", linestyle='-', color='r')
    plt.xlabel(r"$x'$")
    plt.ylabel(r"Amplitude")
    plt.title(r"Cas $n=2$, $m=4$")
    plt.legend()
    plt.grid()

    # Deuxième graphe : n = 3, m = 4
    plt.subplot(1, 2, 2)
    plt.plot(x, sin_n2, label=r"$\sin(3\pi x')$", linestyle='--', color='b')
    plt.plot(x, sin_m, label=r"$\sin(4\pi x')$", linestyle='-.', color='g')
    plt.plot(x, produit2, label=r"$\sin(3\pi x') \sin(4\pi x')$", linestyle='-', color='r')
    plt.xlabel(r"$x'$")
    plt.ylabel(r"Amplitude")
    plt.title(r"Cas $n=3$, $m=4$")
    plt.legend()
    plt.grid()

    plt.show()



# 1.2.2.1 Valeurs propres

def epsilon(N: int) -> np.ndarray:
    """
    IN  >  N (int) : Taille de la matrice Hamiltonienne  
    OUT >  val_propres (numpy.ndarray) : Valeurs propres de la matrice Hamiltonienne  
    """
    H = hamiltonien(N)  # Obtention de la matrice Hamiltonienne
    val_propres = np.linalg.eigh(H)[0]  # Calcul des valeurs propres
    return val_propres  # Retour des valeurs propres


def tracer_epsilon(N: int) -> None:
    """
    IN  >  N (int) : Taille de la matrice Hamiltonienne  
    OUT >  Affiche le graphique des valeurs propres ε en fonction de n  
    """
    n = np.array(range(1, N + 1))  # Indices des niveaux d'énergie
    plt.plot(n, epsilon(N), marker='+', linestyle='None')  # Tracé des valeurs propres
    plt.grid()
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\varepsilon$")
    plt.title(r"$\varepsilon$ = f($n$)")
    plt.show()



# 1.2.2.2 Comparaison des énergies propres 

def E_HO(N: int) -> np.ndarray:
    """
    IN  >  N (int) : Nombre d'états quantiques considérés  
    OUT >  E (numpy.ndarray) : Énergies propres théoriques de l'oscillateur harmonique  
    """
    R = 24  # Paramètre de l'oscillateur harmonique
    n = np.array(range(1, N + 1))  # Indices des états quantiques
    E = R * (n - 1/2)  # Calcul des énergies théoriques
    return E  # Retour des énergies théoriques


def tracer_E_HO(N: int) -> None:
    """
    IN  >  N (int) : Nombre d'états quantiques à tracer  
    OUT >  Affiche le graphique des énergies propres théoriques en fonction de n.  
    """
    n = np.array(range(1, N + 1))  # Indices des états quantiques
    plt.plot(n, E_HO(N), '+')  # Tracé des énergies théoriques
    plt.grid()
    plt.xlabel(r"$n$")
    plt.ylabel(r"$E_{HO}$")
    plt.title(r"$E_{HO}$ = f($n$)")
    plt.show()


def tracer_comparaison(N: int) -> None:
    """
    IN  >  N (int) : Nombre d'états quantiques à tracer  
    OUT >  Affiche le graphique comparant les énergies propres numériques et théoriques.  
    """
    n = np.array(range(1, N + 1))  # Indices des états quantiques
    
    plt.plot(n, epsilon(N), 'o', label=r"$\varepsilon(n)$ (numérique)")  # Tracé des valeurs propres numériques
    plt.plot(n, E_HO(N), '+', label=r"$E_{HO}(n)$ (théorique)")  # Tracé des énergies théoriques

    plt.grid()
    plt.xlabel(r"$n$")
    plt.ylabel(r"Énergie")
    plt.title(r"Comparaison des énergies propres : $\varepsilon(n)$ et $E_{HO}(n)$")
    plt.legend()
    plt.show()



# 1.2.2.3 Comparaison à un modèle quadratique 

def tracer_comparaison_complete(N: int) -> None:
    """
    IN  >  N (int) : Nombre d'états quantiques à tracer  
    OUT >  Affiche le graphique comparant les énergies propres numériques, théoriques et le modèle quadratique.  
    """
    n = np.array(range(1, N + 1))  # Indices des états quantiques

    eps = epsilon(N)  # Énergies propres numériques
    E_HO_values = E_HO(N)  # Énergies théoriques
    C = np.mean(eps - n**2)  # Constante d'ajustement du modèle quadratique
    modele_quadratique = n**2 + C  # Modèle quadratique ajusté

    plt.plot(n, eps, 'o', label=r"$\epsilon(n)$ (numérique)")  # Tracé des énergies numériques
    plt.plot(n, E_HO_values, '+', label=r"$E_{HO}(n)$ (théorique)")  # Tracé des énergies théoriques
    plt.plot(n, modele_quadratique, 'x', label=r"Modèle quadratique")  # Tracé du modèle quadratique

    plt.grid()
    plt.xlabel(r"$n$")
    plt.ylabel(r"Énergie")
    plt.title(r"Comparaison des énergies : Numérique, Théorique et Quadratique")
    plt.legend()
    plt.show()

    return C



# 1.2.3.1 Fonctions d'onde de l'état fondamental

def phi(n: int, x: float) -> float:
    """
    IN  >  n (int) : Indice de l'état propre  
           x (float) : Position dans le puits de potentiel  
    OUT >  phi_n_x (float) : Valeur de la fonction propre phi(n, x)  
    """
    a = 1  # Paramètre du puits de potentiel
    # Calcul de la fonction propre phi(n, x) selon l'équation de l'oscillateur quantique
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

def psi0(N: int, x: float) -> float:
    """
    IN  >  N (int) : Taille de la matrice hamiltonienne (nombre d'états pris en compte)  
           x (float) : Position dans le puits de potentiel  
    OUT >  psi0_x (float) : Approximation numérique de la fonction d'onde fondamentale ψ₀(x)  
    """
    S = 0  # Initialisation de la somme pour l'approximation de la fonction d'onde
    H = hamiltonien(N)  # Calcul de la matrice hamiltonienne pour N états
    vect_propres = np.linalg.eigh(H)[1][:, 0]  # Récupération du premier vecteur propre (état fondamental)
    
    # Approximation de la fonction d'onde ψ₀(x) par une somme des vecteurs propres pondérés
    for m in range(1, N + 1):
        S += vect_propres[m - 1] * phi(m, x)  # Somme des contributions de chaque fonction propre
    
    return S  # Retourne l'approximation de la fonction d'onde fondamentale

def psi_theorique(x: float) -> float:
    """
    IN  >  x (float) : Position dans le puits de potentiel 
    OUT >  psi_th_x (float) : Valeur théorique de la fonction d'onde fondamentale de l'oscillateur harmonique  
    """
    a = 1  # Paramètre du puits de potentiel
    R = 24  # Constante de l'oscillateur harmonique
    pi = np.pi  # Valeur de pi
    # Calcul de la fonction d'onde théorique pour l'état fondamental d'un oscillateur harmonique
    psi_0 = (pi * R / (2 * a**2))**(1/4) * np.exp(-pi**2 / (4 * R) * (x / a - 1 / 2)**2)
    return psi_0  # Retourne la valeur théorique de la fonction d'onde fondamentale
  
def tracer_psi0() -> None:
    """
    IN  >  None (utilise une liste prédéfinie de valeurs de N)  
    OUT >  Affichage du graphe des fonctions d'onde fondamentales approximées et théorique  
    """
    a = 1  # Paramètre du puits de potentiel
    Liste_N = [3, 5, 8, 15]  # Liste des valeurs de N pour lesquelles tracer les fonctions d'onde
    x = np.linspace(0, a, 1000)  # Discrétisation de l'intervalle [0, a]
    
    # Tracé des fonctions d'onde approximées pour chaque valeur de N
    for N in Liste_N:
        if max(psi0(N, x)) < 0.5:  # Condition pour retourner les valeurs négatives si nécessaire
            plt.plot(x, -psi0(N, x), label=f"N={N}")
        else:
            plt.plot(x, psi0(N, x), label=f"N={N}")
    
    plt.plot(x, psi_theorique(x), label=r"Fonction d'onde théorique")  # Tracé de la fonction d'onde théorique
    plt.xlabel(r"$x$")
    plt.ylabel(r"Fonctions d'onde")
    plt.title(r"Fonctions d'onde de l'état fondamental pour $N$=3, 5, 8 ou 15")
    plt.grid()
    plt.legend()
    plt.show()  # Affichage du graphique


# 1.2.3.2 Fonctions d'onde du premier état excité

def psi1(N: int, x: float) -> float:
    """
    IN  >  N (int) : Nombre d'états pris en compte dans l'approximation  
           x (float) : Position dans le puits de potentiel  
    OUT >  psi1_x (float) : Approximation numérique de la fonction d'onde du premier état excité ψ₁(x)  
    """
    S = 0  # Initialisation de la somme pour l'approximation de la fonction d'onde
    H = hamiltonien(N)  # Matrice hamiltonienne pour N états
    vect_propres = np.linalg.eigh(H)[1][:, 1]  # Récupération du second vecteur propre (premier état excité)
    
    # Approximation de la fonction d'onde ψ₁(x) par une somme des vecteurs propres pondérés
    for m in range(1, N + 1):
        S += vect_propres[m - 1] * phi(m, x)  # Somme des contributions de chaque fonction propre
    
    return S  # Retourne l'approximation de la fonction d'onde du premier état excité

def psi_theorique1(x: float) -> float:
    """
    IN  >  x (float) : Position dans le puits de potentiel  
    OUT >  psi1_th_x (float) : Valeur théorique de la fonction d'onde du premier état excité  
    """
    a = 1  # Paramètre du puits de potentiel
    R = 24  # Constante de l'oscillateur harmonique
    pi = np.pi  # Valeur de pi
    # Calcul de la fonction d'onde théorique pour le premier état excité
    psi1_th_x = (pi**5 * R**3 / (2 * a**2))**(1/4) * np.exp(-pi**2 / (4 * R) * (x / a - 1 / 2)**2) * (x / a - 1 / 2)
    return psi1_th_x  # Retourne la valeur théorique de la fonction d'onde du premier état excité


def tracer_psi1() -> None:
    """
    IN  >  None (utilise une liste prédéfinie de valeurs de N)  
    OUT >  Affichage du graphe des fonctions d'onde du premier état excité  
    """
    a = 1  # Paramètre du puits de potentiel
    x = np.linspace(0, a, 100000)  # Discrétisation de l'intervalle [0, a]
    Liste_N = [3, 5, 8, 15]  # Liste des valeurs de N pour lesquelles tracer les fonctions d'onde

    # Tracé des fonctions d'onde approximées pour chaque valeur de N
    for N in Liste_N:
        psi_values = psi1(N, x)  # Calcul des valeurs approximées de la fonction d'onde
        plt.plot(x, -psi_values, label=f"N={N}")  # Tracé de la fonction d'onde pour chaque N

    plt.plot(x, psi_theorique1(x), label="Fonction d'onde théorique", linestyle="dashed", color="black")  # Tracé de la fonction théorique

    plt.xlabel(r"$x$")
    plt.ylabel(r"Fonctions d'onde")
    plt.title(r"Fonctions d'onde du premier état excité pour $N$=3, 5, 8, 15")
    plt.grid()
    plt.legend()
    plt.show()  # Affichage du graphique


# 1.2.3.3 Fonctions d'onde du deuxième état excité

def psi2(N: int, x: float) -> float:
    """
    IN  >  N (int) : Nombre d'états pris en compte dans l'approximation  
           x (float) : Position dans le puits de potentiel  
    OUT >  psi2_x (float) : Approximation numérique de la fonction d'onde du second état excité ψ₂(x)  
    """
    S = 0  # Initialisation de la somme pour l'approximation de la fonction d'onde
    H = hamiltonien(N)  # Matrice hamiltonienne pour N états
    vect_propres = np.linalg.eigh(H)[1][:, 2]  # Récupération du troisième vecteur propre (second état excité)

    # Approximation de la fonction d'onde ψ₂(x) par une somme des vecteurs propres pondérés
    for m in range(1, N + 1):
        S += vect_propres[m - 1] * phi(m, x)  # Somme des contributions de chaque fonction propre

    return S  # Retourne l'approximation de la fonction d'onde du second état excité

def psi_theorique2(x: float) -> float:
    """
    IN  >  x (float) : Position dans le puits de potentiel  
    OUT >  psi2_th_x (float) : Valeur théorique de la fonction d'onde du second état excité  
    """
    a = 1  # Paramètre du puits de potentiel
    R = 24  # Constante de l'oscillateur harmonique
    pi = np.pi  # Valeur de pi
    # Calcul de la fonction d'onde théorique pour le second état excité
    psi2_th_x = (pi * R / (8 * a**2))**(1/4) * np.exp(-pi**2 / (4 * R) * (x / a - 1 / 2)**2) * (pi**2 * R * (x / a - 1 / 2)**2 - 1)
    return psi2_th_x  # Retourne la valeur théorique de la fonction d'onde du second état excité


def tracer_psi2() -> None:
    """
    IN  >  None (utilise une liste prédéfinie de valeurs de N)  
    OUT >  None (Affichage du graphe des fonctions d'onde du second état excité)  
    """
    a = 1  # Paramètre du puits de potentiel
    x = np.linspace(0, a, 100000)  # Discrétisation de l'intervalle [0, a]
    Liste_N = [3, 5, 8, 15]  # Liste des valeurs de N pour lesquelles tracer les fonctions d'onde

    # Tracé des fonctions d'onde approximées pour chaque valeur de N
    for N in Liste_N:
        psi_values = psi2(N, x)  # Calcul des valeurs approximées de la fonction d'onde
        if psi2(N, 0.5) > 10**(-13):  # Condition pour éviter les petites valeurs proches de zéro
            plt.plot(x, -psi_values, label=f"N={N}")
        else:
            plt.plot(x, psi_values, label=f"N={N}")

    plt.plot(x, psi_theorique2(x), label="Fonction d'onde théorique", linestyle="dashed", color="black")  # Tracé de la fonction théorique

    plt.xlabel(r"$x$")
    plt.ylabel(r"Fonctions d'onde")
    plt.title(r"Fonctions d'onde du second état excité pour $N$=3, 5, 8, 15")
    plt.grid()
    plt.legend()
    plt.show()  # Affichage du graphique
