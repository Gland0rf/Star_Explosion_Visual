from button import Button
from data.preprocess_data import Preprocess
import pygame
import numpy as np

# Moffat data
amp = 100
xc = 0
yc = 0
radius = 6
beta = 4.5

x = np.linspace(-20, 20, 1600)
y = np.linspace(-20, 20, 1600)
X, Y = np.meshgrid(x, y)

# Generate Moffat star data for the entire animation
animation_frames = 40 # Can NOT be an odd number

def saveData(name, star_frames, normalized_data, precompiled_surface):
    precompiled_surface_array = [pygame.surfarray.array3d(surface) for surface in precompiled_surface]
    np.savez(name, star_frames=star_frames, normalized_data=normalized_data, precompiled_surface=precompiled_surface_array)
    
def load_data(name):
    loaded_data = np.load(name)
    star_frames = loaded_data["star_frames"]
    normalized_data = loaded_data["normalized_data"]
    precompiled_surface_array = loaded_data["precompiled_surface"]
    precompiled_surface = [pygame.surfarray.make_surface(np.rot90(surface).astype(np.uint8)) for surface in precompiled_surface_array]
    
    return star_frames, normalized_data, precompiled_surface

input_data = -1
print("Re-process or load data? 0 -> Pre-process, 1 -> Load (Type 0 if this is your first time running this file.)")
while input_data != "0" and input_data != "1":
    input_data = input()

if input_data == "0":
    preprocess_data = Preprocess()
    print("Standard Data preprocessing...")
    star_frames, normalized_data, precompiled_surface = preprocess_data.preproccess_Normal_Data(animation_frames, X, Y, amp, xc, yc, radius, beta, 0.1)
    print("\nEnergy Removal Data preprocessing...")
    energy_removal_star_frames, energy_removal_normalized_data, energy_removal_precompiled_surface = preprocess_data.preproccess_AntiEnergy_Data(animation_frames, X, Y, amp, xc, yc, radius, beta, 0.1)
    print("\nExplosion Data preprocessing...")
    exploding_star_frames, exploding_normalized_data, exploding_precompiled_surface = preprocess_data.preproccess_Exploding_Data(animation_frames, X, Y, amp, xc, yc, radius, beta, 1, -0.1)
    saveData("data/preprocessed_normal_data.npz", star_frames, normalized_data, precompiled_surface)
    saveData("data/preprocessed_antienergy_data.npz", energy_removal_star_frames, energy_removal_normalized_data, energy_removal_precompiled_surface)
    saveData("data/preprocessed_exploding_data.npz", exploding_star_frames, exploding_normalized_data, exploding_precompiled_surface)
else:
    print("Loading data...")
    star_frames, normalized_data, precompiled_surface = load_data("data/preprocessed_normal_data.npz")
    energy_removal_star_frames, energy_removal_normalized_data, energy_removal_precompiled_surface = load_data("data/preprocessed_antienergy_data.npz")
    exploding_star_frames, exploding_normalized_data, exploding_precompiled_surface = load_data("data/preprocessed_exploding_data.npz")

# Initialize Pygame
pygame.init()

# Set up Pygame window
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Moffat Star')
clock = pygame.time.Clock()

# Scale the image to fit the entire screen
scale_factor = min(width / X.shape[1], height / X.shape[0])
img_surface = pygame.surfarray.make_surface(np.rot90(star_frames[0]))
img_surface = pygame.transform.scale(img_surface, (int(X.shape[1] * scale_factor), int(X.shape[0] * scale_factor)))

img_rect = img_surface.get_rect(center=(width / 2, height / 2))

#Energy button
energy_button = Button(250, 700, 300, 50, "Remove all energy", (121, 32, 35))

running = True
run_animation = True
run_explosion = False
return_to_beginning_state = False
reached_explosion_end = False
frame = 0

energy_removed = False
surface_used = precompiled_surface

while running and frame < 50:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            energy_button.check_hover(pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if energy_button.check_click(pygame.mouse.get_pos()):
                energy_removed = True
    
    if energy_removed:
        #Change star color to yellow
        yellow_color = np.array([255, 255, 0])
        surface_used = energy_removal_precompiled_surface
        return_to_beginning_state = True
        
        energy_removed = False
    if run_explosion:
        # Activate shake
        surface_used = exploding_precompiled_surface

    surface = pygame.transform.scale(surface_used[frame], (int(X.shape[1] * scale_factor), int(X.shape[0] * scale_factor)))
    if reached_explosion_end:
        surface.fill((255, 255, 255))
    
    # Blit the surface onto the screen
    screen.blit(surface, img_rect)
    
    # Draw button
    energy_button.draw(screen)

    pygame.display.flip()
    clock.tick(30)  # Increase the speed to 60 FPS

    if run_animation:
        if frame < animation_frames - 1:
                frame += 1
        else:
            if return_to_beginning_state:
                return_to_beginning_state = False
                run_explosion = True
            elif run_explosion:
                reached_explosion_end = True
                run_animation = False
                continue
            frame = 0

pygame.quit()