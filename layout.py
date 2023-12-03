import random

# 🟦
# 🟩
# 🟥
# 🟨
# ⬜

class creatingLayout():
    def __init__(self):
        self.grid = None
        self.color = ['🟦', '🟩', '🟥', '🟨']
        self.picked_rows = []
        self.picked_cols = []
        self.debug = False
        self.color_combinations = []
        # self.wiredGrid(D)

    def wiredGrid(self, D):
        self.grid = [["⬜️" for _ in range(D)] for _ in range(D)]
        self.color_combinations = []
        which = random.choice([True, False])
        if which:
            while len(self.color) > 0:
                self.getRow(D)
                self.getCol(D)
        else:
            while len(self.color) > 0:
                self.getCol(D)
                self.getRow(D)

        # Print Grid
        if self.debug:
            for x in self.grid:
                print(''.join(x))
            print()

        print(self.color_combinations)
        return self.grid #, self.wiredGrid_status_is

    def getRow(self, D):
        pick_row = random.randint(0, D - 1)
        while pick_row in self.picked_rows:
            pick_row = random.randint(0, D - 1)
        self.picked_rows.append(pick_row)
        pick_color = random.randint(0, len(self.color) - 1)
        for y in range(D):
            self.grid[pick_row][y] = self.color[pick_color]
        self.color_combinations.append((self.color[pick_color], 0))  # 0 represents row
        self.color.pop(pick_color)

    def getCol(self, D):
        pick_col = random.randint(0, D - 1)
        while pick_col in self.picked_cols:
            pick_col = random.randint(0, D - 1)
        self.picked_cols.append(pick_col)
        pick_color = random.randint(0, len(self.color) - 1)
        for x in range(D):
            self.grid[x][pick_col] = self.color[pick_color]
        self.color_combinations.append((self.color[pick_color], 1))  # 1 represents column
        self.color.pop(pick_color)

    def wiredGrid_status_is(self):
        redPixel_index = self.find_indexes(self.color_combinations, "🟥")
        yellowPixel_index = self.find_indexes(self.color_combinations, "🟨")
        if yellowPixel_index > redPixel_index:
            return 1  # Danger
        elif yellowPixel_index < redPixel_index:
            return 0  # Safe

    def find_indexes(self, givenList, target_color):
        indexes = [index for index, (color, _) in enumerate(givenList) if color == target_color]
        return indexes[0]
