# Contexte du projet
L’Organisation nationale de lutte contre le faux-monnayage (ONCFM) est une organisation publique ayant pour objectif de mettre en place des méthodes d’identification des contrefaçons des billets en euros. Dans le cadre de cette lutte, nous souhaitons mettre en place un algorithme qui soit capable de différencier automatiquement les vrais des faux billets.

# Objectifs
Lorsqu’un billet arrive, nous avons une machine qui consigne l’ensemble de ses caractéristiques géométriques. Au fil de nos années de lutte, nous avons observé des différences de dimensions entre les vrais et les faux billets. Ces différences sont difficilement visibles à l’oeil nu, mais une machine devrait sans problème arriver à les différencier.
Ainsi, il faudrait construire un algorithme qui, à partir des caractéristiques géométriques d’un billet, serait capable de définir si ce dernier est un vrai ou un faux billet.

# Modèle de données

## Dimensions géométriques
Nous disposons actuellement de six informations géométriques sur un billet :
- length : la longueur du billet (en mm) ;
- height_left : la hauteur du billet (mesurée sur le côté gauche, en mm) ;
- height_right : la hauteur du billet (mesurée sur le côté droit, en mm) ;
- margin_up : la marge entre le bord supérieur du billet et l'image de celui-ci (en mm) ;
- margin_low : la marge entre le bord inférieur du billet et l'image de celui-ci (en mm) ;
- diagonal : la diagonale du billet (en mm).
Ces informations sont celles avec lesquelles l’algorithme devra opérer.

# Fonctionnement général
Comme vu précédemment, nous avons à notre disposition six données géométriques pour chaque billet. L’algorithme devra donc être capable de
prendre en entrée un fichier contenant les dimensions de plusieurs billets, et de déterminer le type de chacun d’entre eux à partir des seules dimensions.

Nous aimerions pouvoir mettre en concurrence quatre méthodes de prédiction :
- une régression logistique classique ;
- un k-means, duquel seront utilisés les centroïdes pour réaliser la prédiction ;
- un KNN ;
- un random forest.

Cet algorithme se devra d’être naturellement le plus performant possible pour identifier un maximum de faux billets au sein de la masse de
billets analysés chaque jour.
