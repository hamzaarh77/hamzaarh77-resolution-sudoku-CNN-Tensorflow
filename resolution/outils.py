import numpy as np



def normaliser(a):    
    return (a/9)-.5


def denormaliser(a):
    return (a+.5)*9




def correspondance(x, y):
    cpt = np.sum(x == y)
    percentage = (cpt / np.prod(x.shape)) * 100
    print("percentage de correspondance :",str(percentage),"%")


#fonction de résolution
def solver(grille,SudokuCNN):

    while(1):
    
        out = SudokuCNN.predict(grille.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        grille = denormaliser(grille).reshape((9,9))
        mask = (grille==0) # mask est un tableau de bool ou les cases vide sont representer par true
     
        if(mask.sum()==0): # dans le cas ou il y a pas de case vide
            break
            
        prob2 = prob*mask # conserve les probabilité lié au cases vides
    
        index = np.argmax(prob2) #indice de la case vide avec la plus haute probabilité de solution parmis toute les cases vide de la grille
        x, y = divmod(index,9)

        val = pred[x][y] # val est la valeur predite pour cette case 
        grille[x][y] = val
        grille = normaliser(grille)
    
    return pred


# fonction principale
# doit transformer une matrice 9x9 en  array 9,9,1 en suite le le normalize pour le donne au solver
def solve_sudoku(game,SudokuCNN):
    
    game = np.array([j for j in game]).reshape((9,9,1)) 
    game = normaliser(game)
    game = solver(game,SudokuCNN)
    return game