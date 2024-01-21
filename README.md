# Classification d'images de lettres avec des algorithmes de Machine Learning

## Présentation

L'objectif de ce projet est de créer un modèle de machine learning capable de déterminer la lettre présentée sur une image. Nous allons utiliser un réseau de neurones convolutif (CNN) pour résoudre ce problème. Pour cela, nous avons à disposition un jeu de données 'notMNIST' qui contient 10 dossiers pour 10 classes de lettres (A à J) à prédire. Le modèle sera entraîné pour reconnaître uniquement les lettres allant de A à J. Pour chaque lettre, nous disposons de 1872 images, que l'on répartira entre les données d'entraînement, de validation et de test. Les images ont une taille de 28x28 pixels et sont en niveau de gris (1 canal).

Pour estimer la performance du modèle, nous utiliserons la précision (accuracy) comme métrique. La précision est le nombre de prédictions correctes divisé par le nombre total de prédictions. Elle est utilisée pour les problèmes de classification où le nombre de classes est équilibré. Dans notre cas, nous avons 10 classes et la répartition est la même entre ces 10 classes. Nous sommes également sensibles à la fonction de perte (loss) qui correspond à l'écart entre les prévisions du modèle et les observations réelles. Le but est de minimiser cette fonction de perte.

Pour le chargement des données, la construction du modèle, son entraînement, l'évaluation des résultats et l'affichage graphique des résultats, nous utilisons différentes librairies Python. Nous utlisons la librairie matplotlib pour le chargement des images et l'affichage de courbes de résultat. Keras et TensorFlow permettent la construction du modèle en différentes couches, l'entraînement et l'évaluation des résultats. Nous utilisons également la librairie numpy pour la manipulation des données, sklearn pour créer la matrice de confusion et les librairies glob et os pour la gestion des fichiers.

Nous avons également essayé d'effectuer du transfert learning en utilisant le modèle pré-entraîné MobileNetV2. Il est plus léger que VGG16, mais nous devons tout de même ruser pour lui fournir des images en 3 canaux alors que nous n'en avons qu'un seul. Au final le résultat satisfaisant, l'accurary est inférieure à celle obtenue avec le CNN.

## Prérequis

Pour que le notebook fonctionne correctement, il faut que les librairies à importer soient installées. Il faut également se placer au même niveau que le dossier 'data' contenant le dossier 'notMNIST_small' qui contient les images à charger réparties en 10 dossiers (un par classe) portant le nom de la lettre correspondante.
Si vous souhaitez tenir compte des données augmentées (générées avec le script 'gan_data_augmentation.py'), il faut vous assurer que le dossier 'data' contient également le dossier 'notMNIST_small_augmented' qui contient les images générées réparties en 10 dossiers (un par classe) portant le nom de la lettre correspondante.

## Auteurs

- [Maxime Dupont](https://github.com/maxime-dupont01)
- [Adrien Desbrosses](https://github.com/bvzopa)
