import random
import AI_Learn
import pandas as pd

# 🟦
# 🟩
# 🟥
# 🟨
# ⬜

# creating a class where images will be generated
class creatingLayout():
    # init method to initialize all the variable
    def __init__(self):
        self.grid = None
        self.color = ['🟦', '🟩', '🟥', '🟨']
        self.picked_rows = []
        self.picked_cols = []
        self.debug = False
        self.color_combinations = []

    # Method to place the wires on on the image
    def wiredGrid(self, D):
        # create grid
        self.grid = [["⬜" for _ in range(D)] for _ in range(D)]
        # dictionry to store the combination
        self.color_combinations = []
        # Ask 50% chance to Pick row or col
        which = random.choice([True, False])
        
        # If which is True then row else coln
        if which:
            # Loop until all four wires places on image
            while len(self.color) > 0:
                # first row then coln
                self.getRow(D)
                self.getCol(D)
        else:
            # Loop until all four wires places on image
            while len(self.color) > 0:
                # first coln then row
                self.getCol(D)
                self.getRow(D)

        # Print Grid when debug mode is ON
        if self.debug:
            for x in self.grid:
                print(''.join(x))
            print()
            
        return self.grid , self.color_combinations

    # Get ROW
    def getRow(self, D):
        # Pick a random row from the grid
        pick_row = random.randint(0, D - 1)
        # Check if the row pick from the grid is not same as picked last time
        while pick_row in self.picked_rows:
            pick_row = random.randint(0, D - 1)
        self.picked_rows.append(pick_row)
        # Pick a random color for the row
        pick_color = random.randint(0, len(self.color) - 1)
        # Iterate through the row and place the wire of choosen color
        for y in range(D):
            self.grid[pick_row][y] = self.color[pick_color]
        # order of the color is appended to the color combination
        self.color_combinations.append((self.color[pick_color], 0))  # 0 represents row
        # remove the picked color from the color variable
        self.color.pop(pick_color)

    # Get COLUMN
    def getCol(self, D):
        # Pick a random column from the grid
        pick_col = random.randint(0, D - 1)
        # Check if the column is picked from the grid is not same as picked last time
        while pick_col in self.picked_cols:
            pick_col = random.randint(0, D - 1)
        self.picked_cols.append(pick_col)
        # Pick a random color for the columns
        pick_color = random.randint(0, len(self.color) - 1)
        # Iterate through the column and place the wire of choosen color
        for x in range(D):
            self.grid[x][pick_col] = self.color[pick_color]
        # order of the color is appended to the color combination
        self.color_combinations.append((self.color[pick_color], 1))  # 1 represents column
        # remove the picked color from the color variable
        self.color.pop(pick_color)

    # Wired Image status
    def wiredGrid_status_is(self):
        # Get the red wire index from the color combination dictionary
        redPixel_index = self.find_indexes(self.color_combinations, "🟥")
        # Get the yellow wire index from the color combination dictionary
        yellowPixel_index = self.find_indexes(self.color_combinations, "🟨")
        # If the yellow wire index is greater than red means the yellow is placed after red
        if yellowPixel_index > redPixel_index:
            return 1  # Danger
        # If the yellow wire index is smaller than red means the yellow is placed before red
        elif yellowPixel_index < redPixel_index:
            return 0  # Safe

    # Helper Method to find the index from list with the target component
    def find_indexes(self, givenList, target_color):
        indexes = [index for index, (color, _) in enumerate(givenList) if color == target_color]
        return indexes[0]

# Generate data:- 
# Parameters are dataset sixe, grid size, and test_split_ratio
def generate_data_for_task_1(dataSet , gridSize, test_split_ratio):
    result_list = []
    try :
        for i in range(dataSet):
            # Create a layout
            layout = creatingLayout()
            
            # get the grid and its order of wire places
            grid , color_order = layout.wiredGrid(gridSize)

            # Get Circuit status whether its dangerous or nor
            circuit_status = layout.wiredGrid_status_is()
            
            # result dictionary contains the columns names:
            # 1) encoding - represents the hot encoding of the each pixel flatten into 1D array
            # 2) binary classification - represents the binary 1 for danger and 0 for safe
            # 3) multi classification - represents the color at the third position using hot encoding
            result_dict = {'encoding': AI_Learn.encoding_for_1(grid), 
                           'binary_classification': circuit_status }
            
            # Enter the result for from the dictionary to the result list
            result_list.append(result_dict)
            
            # if i == 500:
            #     result_df = pd.DataFrame(result_list)
            #     result_df.to_excel('DataSets/validation_dataSet_1.xlsx', index=False)
            #     result_list = []
                
            # Based on the set training ratio, data will be save to training_dataSet
            if i == dataSet - int((dataSet * test_split_ratio)):
                result_df = pd.DataFrame(result_list)
                result_df.to_excel('DataSets/training_dataSet_1.xlsx', index=False)
                # After entering storing to training result list is empty to store new data to test
                result_list = []
            
            # Based on the set test ratio, data will be save to training_dataSet
            if i == dataSet - 1:
                result_df = pd.DataFrame(result_list)
                result_df.to_excel('DataSets/testing_dataSet_1.xlsx', index=False)
                
                
    # Any exception occured for the file opening
    except Exception as e:
        print("Please make sure for the file path. \n Error", e)
        
    print("***********************************************************")
    print("  ->   DATA GENERATED FOR MODEL - 1  <-   ")
    print("***********************************************************")


# Parameters are dataset sixe, grid size, and test_split_ratio
def generate_data_for_task_2(dataSet , gridSize, test_split_ratio):
    result_list = []
    va = 0
    check = 0
    try :
        while check != 2:
            # Create a layout
            layout = creatingLayout()
            # get the grid and its order of wire places
            grid , color_order = layout.wiredGrid(gridSize)
            
            # Get Circuit status whether its dangerous or nor
            circuit_status = layout.wiredGrid_status_is()
            
            if circuit_status == 1:

                result_dict = {'encoding': AI_Learn.encoding_for_2(grid), 
                            'multi_classification': AI_Learn.order_encoding(color_order) }
                result_list.append(result_dict)
                
                if va == (dataSet - int((dataSet * test_split_ratio))):
                    result_df = pd.DataFrame(result_list)
                    result_df.to_excel('DataSets/training_dataSet_2.xlsx', index=False)
                    check += 1
                
                # if va == 500:
                #     result_df = pd.DataFrame(result_list)
                #     result_df.to_excel('DataSets/validation_dataSet_2.xlsx', index=False)
                
                if va == int((dataSet * test_split_ratio)):
                    result_df = pd.DataFrame(result_list)
                    result_df.to_excel('DataSets/testing_dataSet_2.xlsx', index=False)
                    result_list = []
                    check += 1 
                    
                va += 1
                        
        # Any exception occured for the file opening
    except Exception as e:
        print("Please make sure for the file path. \n Error", e)
        
    print("***********************************************************")
    print("  ->   DATA GENERATED FOR MODEL - 2  <-   ")
    print("***********************************************************")