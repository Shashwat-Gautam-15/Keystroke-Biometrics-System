# This includes multiline 
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# Sample keyboard layout (positions of QWERTY keys)
# QWERTY Keyboard layout data


key_positions = {
    # Number row
    '`': {'pos': (0, 4), 'start': 'a'},
    '1': {'pos': (1, 4), 'start': 'a'},
    '2': {'pos': (2, 4), 'start': 'a'},
    '3': {'pos': (3, 4), 'start': 's'},
    '4': {'pos': (4, 4), 'start': 'd'},
    '5': {'pos': (5, 4), 'start': 'f'},
    '6': {'pos': (6, 4), 'start': 'j'},
    '7': {'pos': (7, 4), 'start': 'j'},
    '8': {'pos': (8, 4), 'start': 'k'},
    '9': {'pos': (9, 4), 'start': 'l'},
    '0': {'pos': (10, 4), 'start': ';'},
    '-': {'pos': (11, 4), 'start': ';'},
    '=': {'pos': (12, 4), 'start': ';'},
    
    # Top letter row
    'q': {'pos': (1.5, 3), 'start': 'a'},
    'w': {'pos': (2.5, 3), 'start': 's'},
    'e': {'pos': (3.5, 3), 'start': 'd'},
    'r': {'pos': (4.5, 3), 'start': 'f'},
    't': {'pos': (5.5, 3), 'start': 'f'},
    'y': {'pos': (6.5, 3), 'start': 'j'},
    'u': {'pos': (7.5, 3), 'start': 'j'},
    'i': {'pos': (8.5, 3), 'start': 'k'},
    'o': {'pos': (9.5, 3), 'start': 'l'},
    'p': {'pos': (10.5, 3), 'start': ';'},
    '[': {'pos': (11.5, 3), 'start': ';'},
    ']': {'pos': (12.5, 3), 'start': ';'},
    '\\': {'pos': (13.5, 3), 'start': ';'},
    
    # Home row
    'a': {'pos': (1.75, 2), 'start': 'a'},
    's': {'pos': (2.75, 2), 'start': 's'},
    'd': {'pos': (3.75, 2), 'start': 'd'},
    'f': {'pos': (4.75, 2), 'start': 'f'},
    'g': {'pos': (5.75, 2), 'start': 'f'},
    'h': {'pos': (6.75, 2), 'start': 'j'},
    'j': {'pos': (7.75, 2), 'start': 'j'},
    'k': {'pos': (8.75, 2), 'start': 'k'},
    'l': {'pos': (9.75, 2), 'start': 'l'},
    ';': {'pos': (10.75, 2), 'start': ';'},
    "'": {'pos': (11.75, 2), 'start': ';'},
    
    # Bottom letter row
    'z': {'pos': (2.25, 1), 'start': 'a'},
    'x': {'pos': (3.25, 1), 'start': 's'},
    'c': {'pos': (4.25, 1), 'start': 'd'},
    'v': {'pos': (5.25, 1), 'start': 'f'},
    'b': {'pos': (6.25, 1), 'start': 'f'},
    'n': {'pos': (7.25, 1), 'start': 'j'},
    'm': {'pos': (8.25, 1), 'start': 'j'},
    ',': {'pos': (9.25, 1), 'start': 'k'},
    '.': {'pos': (10.25, 1), 'start': 'l'},
    '/': {'pos': (11.25, 1), 'start': ';'},
    
    # Special keys
    'Shift_L': {'pos': (0, 1), 'start': 'a'},
    'Shift_R': {'pos': (12.5, 1), 'start': ';'},
    'Ctrl_L': {'pos': (0, 0), 'start': 'a'},
    'Alt_L': {'pos': (2, 0), 'start': 'a'},
    'Space': {'pos': (5, 0), 'start': 'f'},
    'Alt_R': {'pos': (8, 0), 'start': 'j'},
    'Ctrl_R': {'pos': (10, 0), 'start': ';'},
    }

characters = {
    # Lowercase letters (unchanged)
    'a': ('a',), 'b': ('b',), 'c': ('c',), 'd': ('d',), 'e': ('e',),
    'f': ('f',), 'g': ('g',), 'h': ('h',), 'i': ('i',), 'j': ('j',),
    'k': ('k',), 'l': ('l',), 'm': ('m',), 'n': ('n',), 'o': ('o',),
    'p': ('p',), 'q': ('q',), 'r': ('r',), 's': ('s',), 't': ('t',),
    'u': ('u',), 'v': ('v',), 'w': ('w',), 'x': ('x',), 'y': ('y',),
    'z': ('z',),
    
    # Uppercase letters (updated)
    'A': ('Shift_R', 'a'), 'B': ('Shift_R', 'b'), 'C': ('Shift_R', 'c'),
    'D': ('Shift_R', 'd'), 'E': ('Shift_R', 'e'), 'F': ('Shift_R', 'f'),
    'G': ('Shift_R', 'g'), 'H': ('Shift_L', 'h'), 'I': ('Shift_L', 'i'),
    'J': ('Shift_L', 'j'), 'K': ('Shift_L', 'k'), 'L': ('Shift_L', 'l'),
    'M': ('Shift_L', 'm'), 'N': ('Shift_L', 'n'), 'O': ('Shift_L', 'o'),
    'P': ('Shift_L', 'p'), 'Q': ('Shift_R', 'q'), 'R': ('Shift_R', 'r'),
    'S': ('Shift_R', 's'), 'T': ('Shift_R', 't'), 'U': ('Shift_L', 'u'),
    'V': ('Shift_R', 'v'), 'W': ('Shift_R', 'w'), 'X': ('Shift_R', 'x'),
    'Y': ('Shift_L', 'y'), 'Z': ('Shift_R', 'z'),
    
    # Numbers and their shifted symbols (updated)
    '1': ('1',), '!': ('Shift_R', '1'),
    '2': ('2',), '@': ('Shift_R', '2'),
    '3': ('3',), '#': ('Shift_R', '3'),
    '4': ('4',), '$': ('Shift_R', '4'),
    '5': ('5',), '%': ('Shift_R', '5'),
    '6': ('6',), '^': ('Shift_L', '6'),
    '7': ('7',), '&': ('Shift_L', '7'),
    '8': ('8',), '*': ('Shift_L', '8'),
    '9': ('9',), '(': ('Shift_L', '9'),
    '0': ('0',), ')': ('Shift_L', '0'),
    
    # Other symbols (updated)
    '`': ('`',), '~': ('Shift_R', '`'),
    '-': ('-',), '_': ('Shift_L', '-'),
    '=': ('=',), '+': ('Shift_L', '='),
    '[': ('[',), '{': ('Shift_L', '['),
    ']': (']',), '}': ('Shift_L', ']'),
    '\\': ('\\',), '|': ('Shift_L', '\\'),
    ';': (';',), ':': ('Shift_L', ';'),
    "'": ("'",), '"': ('Shift_L', "'"),
    ',': (',',), '<': ('Shift_L', ','),
    '.': ('.',), '>': ('Shift_L', '.'),
    '/': ('/',), '?': ('Shift_L', '/'),
    
    # Space (unchanged)
    ' ': ('Space',),
    }


keys = list(key_positions.keys())

# Generate random keyboard layout
def generate_layout(keys):
    layout = keys.copy()
    random.shuffle(layout)
    return layout

# Distance calcuation via start key
def calculate_distance(text, layout, positions):
    layout_map = {layout[i]: positions[keys[i]]['pos'] for i in range(len(layout))}
    total_distance = 0.0

    for i in range(1, len(text)):
        char_1, char_2 = text[i-1], text[i]
        if char_1 in characters and char_2 in characters:
            keys_1 = characters[char_1]
            keys_2 = characters[char_2]

            for k1 in keys_1:
                for k2 in keys_2:
                    if k1 in layout_map and k2 in layout_map:
                        x1, y1 = layout_map[k1]
                        x2, y2 = layout_map[k2]
                        start_key_1 = positions[k1]['start']
                        start_key_2 = positions[k2]['start']
                        start_pos_1 = layout_map[start_key_1]
                        start_pos_2 = layout_map[start_key_2]

                        # Calculate distances using Euclidean formula
                        dist_1 = np.sqrt((x1 - start_pos_1[0])**2 + (y1 - start_pos_1[1])**2)
                        dist_2 = np.sqrt((x2 - start_pos_2[0])**2 + (y2 - start_pos_2[1])**2)

                        # Adding the total distance i.e from one key to another
                        total_distance += dist_1 + dist_2
    return total_distance


# Generate a neighbor layout solution by swapping 2 or 3 keys
def get_neighbour(current_layout):
    new_layout = current_layout.copy()
    swap_type = random.choice([2, 3])
    indices = random.sample(range(len(current_layout)), swap_type)
    if swap_type == 2:
        i, j = indices
        new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
    else:
        i, j, k = indices
        new_layout[i], new_layout[j], new_layout[k] = new_layout[k], new_layout[i], new_layout[j]
    return new_layout

# Simulated annealing algorithm same as sales man probelm
def simulated_annealing(text, keys, positions, initial_temp, cooling_rate, num_iterations):
    current_layout = generate_layout(keys)
    best_layout = current_layout.copy()

    current_distance = calculate_distance(text, current_layout, positions)
    best_distance = current_distance

    temp = initial_temp
    distances = [current_distance]
    best_layouts = [best_layout.copy()]
    best_distances = [best_distance]

    for i in range(num_iterations):
        neighbour_layout = get_neighbour(current_layout)
        neighbour_distance = calculate_distance(text, neighbour_layout, positions)

        if neighbour_distance < current_distance:
            current_layout = neighbour_layout
            current_distance = neighbour_distance
        else:
            p = np.exp((current_distance - neighbour_distance) / (temp + 1e-6))
            if random.uniform(0,1) < p:
                current_layout = neighbour_layout
                current_distance = neighbour_distance

        if current_distance < best_distance:
            best_layout = current_layout.copy()
            best_distance = current_distance

        temp *= cooling_rate 
        distances.append(current_distance)
        best_layouts.append(best_layout.copy())
        best_distances.append(best_distance)

        if i % 100 == 0:
            print(f"Iteration {i}: Current Distance = {current_distance:.4f}, Best Distance = {best_distance:.4f}")

    return best_layouts, best_distances, distances

# Updating layout and distance
def update_keyboard(frame, keys, positions, layouts, distances, cur_dist, layout_line, distance_line, cur_dist_line):
    layout = layouts[frame]
    coords = np.array([positions[key]['pos'] for key in layout])
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    
    layout_line.set_data(coords[:, 0], coords[:, 1])
    distance_line.set_data(range(frame + 1), distances[:frame + 1])
    cur_dist_line.set_data(range(frame + 1), cur_dist[:frame + 1])
    return layout_line, distance_line, cur_dist_line

# After 1000 iterations the best optimized version keys are stored in this
def store_optimized_keys(layout):
    OPTIMIZED_KEYS = [[] for _ in range(5)]  # 5 rows for the keyboard layout
    added_keys = set() # this is to avoid to duplicate like basically spacebar is stored in row 4 susal location but if multiple spaces in input i was geeting 2 space bar
    # Fill the OPTIMIZED_KEYS based on the layout
    for key in layout:
        if key in added_keys:
            continue
        if key in ['Backspace', 'Tab', 'Caps Lock', 'Enter', 'Shift', 'Control', 'Win', 'Alt', 'Space', 'Menu']:
            # Special keys need to be handled based on their layout position
            if key == 'Backspace':
                OPTIMIZED_KEYS[0].append(key)
            elif key == 'Tab':
                OPTIMIZED_KEYS[1].insert(0, key)
            elif key == 'Caps Lock':
                OPTIMIZED_KEYS[2].insert(0, key)
            elif key == 'Enter':
                OPTIMIZED_KEYS[2].append(key)
            elif key == 'Shift':
                OPTIMIZED_KEYS[3].insert(0, key)
            elif key == 'Control':
                OPTIMIZED_KEYS[4].insert(0, key)
            elif key == 'Space':
                OPTIMIZED_KEYS[4].append(key)
            else:
                continue  # special keys skipped since that are already added

        # Regular keys
        for i, row in enumerate(OPTIMIZED_KEYS):
            if len(row) < 15:  # Ensure each row can have a maximum of 15 keys
                row.append(key)
                break  # Move to the next key

    return OPTIMIZED_KEYS

# Main function to run 
def main():
    # Asking user to input multiline string
    print("Enter text to optimize the keyboard layout for (type 'END' to finish):")
    input_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        input_lines.append(line)
    
    text = " ".join(input_lines)
    
    # Simulated annealing parameters
    initial_temp = 100
    cooling_rate = 0.99
    num_iterations = 1000

    # Running the optimization
    best_layouts, best_distances, cur_dist = simulated_annealing(
        text, keys, key_positions, initial_temp, cooling_rate, num_iterations
    )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Simulated Annealing for Keyboard Layout Optimization (Improved)")

    # Keyboard layout plot
    ax1.set_xlim(-0.5, 10)
    ax1.set_ylim(-0.5, 5)
    ax1.set_title("Best Keyboard Layout")
    layout_line, = ax1.plot([], [], 'o-', color='orange')

    # Typing distance plot
    ax2.set_xlim(0, num_iterations)
    ax2.set_ylim(min(best_distances) * 0.9, max(best_distances) * 1.1)
    ax2.set_title("Typing Distance over Iterations")
    distance_line, = ax2.plot([], [], 'r-')
    cur_dist_line, = ax2.plot([], [], 'g-')
    
    #Storing the optimized keys and making keyboard using it
    optimized_keys = store_optimized_keys(best_layouts[-1])

    # sizes of special keys  and draw keyboard directly from assign 4
    key_sizes = {
        'Backspace': 2, 'Tab': 2, 'Caps Lock': 2, 'Enter': 2, 'Shift': 2,
        'Control': 1.25, 'Win': 1.25, 'Alt': 1.25, 'Space': 2, 'Menu': 1.25
    }
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(16, 6))
    # Define the size of each key (default)
    key_width = 1.0
    key_height = 1.0
    # Function to draw the keyboard
    def draw_keyboard(ax,OPTIMIZED_KEYS):
        y_start = 4
        row_offsets = [0.0, 0.0, 0.0, 0.0, 0.0]  # Adjust this for row offsets if necessary
        # Iterate over the rows of keys
        for row_idx, row in enumerate(OPTIMIZED_KEYS):
            x_start = row_offsets[row_idx]  # Apply the row offset
            y_pos = y_start - row_idx * key_height
            # Iterate over each key in the row
            for key_idx, key in enumerate(row):
                if key == '':
                    continue
                # Determine key size (use default if not specified)
                width = key_sizes.get(key, key_width)
                x_pos = x_start + sum(key_sizes.get(k, key_width) for k in row[:key_idx])
                # Draw a rectangle for each key
                rect = Rectangle((x_pos, y_pos), width, key_height, fill=True, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
            # Add the text for the key
                ax.text(x_pos + width / 2, y_pos + key_height / 2, key, fontsize=12, ha='center', va='center')

    draw_keyboard(ax,optimized_keys)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')  



    # Create the animation
    anim = FuncAnimation(fig, update_keyboard, frames=range(0, num_iterations, 1),
                        fargs=(keys, key_positions, best_layouts, best_distances, cur_dist, layout_line, distance_line, cur_dist_line),
                        interval=50, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

    return anim

if __name__ == "__main__":
    main()
