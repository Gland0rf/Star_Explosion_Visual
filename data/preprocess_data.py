import numpy as np
import pygame

class Preprocess:
    
    def __init__(self):
        # Colors
        self.colors = [(0, (0, 0, 0)), (0.2, (186, 51, 54)), (1, (255, 255, 0))]
        self.no_energy_color = [(0.08, (0, 0, 0)), (0.2, (255, 255, 255))]
        # Interpolate colors for the colormap
        self.colormap = np.linspace(0, 1, 256)
        self.r_interp = np.interp(self.colormap, [point[0] for point in self.colors], [point[1][0] for point in self.colors])
        self.g_interp = np.interp(self.colormap, [point[0] for point in self.colors], [point[1][1] for point in self.colors])
        self.b_interp = np.interp(self.colormap, [point[0] for point in self.colors], [point[1][2] for point in self.colors])

        # Prefefine color for the energy removal event
        self.removal_colormap = np.linspace(0, 1, 256)
        self.removal_r_interp = np.interp(self.removal_colormap, [point[0] for point in self.no_energy_color], [point[1][0] for point in self.no_energy_color])
        self.removal_g_interp = np.interp(self.removal_colormap, [point[0] for point in self.no_energy_color], [point[1][1] for point in self.no_energy_color])
        self.removal_b_interp = np.interp(self.removal_colormap, [point[0] for point in self.no_energy_color], [point[1][2] for point in self.no_energy_color])
    
    def moffat(self, x, y, amp, xc, yc, alpha, beta):
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        return amp * (1 + r ** 2 / alpha**2) ** (-beta)
    
    def preproccess_Normal_Data(self, frames, X, Y, amp, xc, yc, radius, beta, delta):
        animation_frames = frames
        end_iter = -1
        end_value = -1
        
        star_frames = []
        # Precompute frames
        for i in range(int(animation_frames / 2)):
            star_frames.append(self.moffat(X, Y, amp, xc, yc, radius + delta * i, beta))
            print(f"\rCooking animation: {i+1}/{animation_frames}", end="", flush=True)
            if i == animation_frames / 2 - 1:
                end_iter = i + 1
                end_value = delta * i
        print(" Done")
        for i in range(0, -int(animation_frames / 2), -1):
            star_frames.append(self.moffat(X, Y, amp, xc, yc, (radius + end_value) + delta * i, beta))
            print(f"\rCooking animation 2: {abs(i)+1+end_iter}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute the normalized data
        normalized_data = []
        for frame in range(animation_frames):
            normalized_data.append((star_frames[frame] - np.min(star_frames[frame])) / (np.max(star_frames[frame]) - np.min(star_frames[frame])))
            print(f"\rCooking normalized data: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute surface
        precompiled_surface = []
        for frame in range(animation_frames):
            precompiled_surface.append(pygame.surfarray.make_surface(np.rot90(np.stack((
                    np.interp(normalized_data[frame], self.colormap, self.r_interp),
                    np.interp(normalized_data[frame], self.colormap, self.g_interp),
                    np.interp(normalized_data[frame], self.colormap, self.b_interp),
            ), axis=-1)).astype(np.uint8)))
            print(f"\rCooking surface: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")
        
        return star_frames, normalized_data, precompiled_surface

    def preproccess_AntiEnergy_Data(self, frames, X, Y, amp, xc, yc, radius, beta, delta):
        end_iter = -1
        end_value = -1
        animation_frames = frames
        
        star_frames = []
        # Precompute frames
        for i in range(int(animation_frames / 2)):
            star_frames.append(self.moffat(X, Y, amp, xc, yc, radius + delta * i, beta))
            print(f"\rCooking animation: {i+1}/{animation_frames}", end="", flush=True)
            if i == animation_frames / 2 - 1:
                end_iter = i + 1
                end_value = delta * i
        print(" Done")
        for i in range(0, -int(animation_frames / 2), -1):
            star_frames.append(self.moffat(X, Y, amp, xc, yc, (radius + end_value) + delta * i, beta))
            print(f"\rCooking animation 2: {abs(i)+1+end_iter}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute the normalized data
        normalized_data = []
        for frame in range(animation_frames):
            normalized_data.append((star_frames[frame] - np.min(star_frames[frame])) / (np.max(star_frames[frame]) - np.min(star_frames[frame])))
            print(f"\rCooking normalized data: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute surface
        precompiled_surface = []
        for frame in range(animation_frames):
            precompiled_surface.append(pygame.surfarray.make_surface(np.rot90(np.stack((
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_r_interp),
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_g_interp),
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_b_interp),
            ), axis=-1)).astype(np.uint8)))
            print(f"\rCooking surface: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")
        
        return star_frames, normalized_data, precompiled_surface
    
    def preproccess_Exploding_Data(self, frames, X, Y, amp, xc, yc, radius, beta, deltaShake, deltaDecrease):
        end_iter = -1
        end_value = -1
        animation_frames = frames
        
        star_frames = []
        # Precompute frames
        current = 1
        end_frame = int(animation_frames / 2)
        for i in range(end_frame):
            if current == 1:
                xc += deltaShake
            elif current == 2:
                xc -= deltaShake
            elif current == 3:
                yc += deltaShake
            elif current == 4:
                yc -= deltaShake
            
            if current == 4:
                current = 1
            else:
                current += 1
            
            star_frames.append(self.moffat(X, Y, amp, xc, yc, radius + deltaDecrease * i, beta))
            print(f"\rCooking animation: {i+1}/{animation_frames}", end="", flush=True)
            if i == animation_frames / 2 - 1:
                end_iter = i + 1
                end_value = deltaDecrease * i
        print(" Done")
        for i in range(end_frame, animation_frames):
            i -= end_frame
            star_frames.append(self.moffat(X, Y, amp, xc, yc, (radius + end_value) + (i/10) ** (1+(i/4)), beta))
            print(f"\rCooking animation 2: {abs(i)+1+end_iter}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute the normalized data
        normalized_data = []
        for frame in range(animation_frames):
            normalized_data.append((star_frames[frame] - np.min(star_frames[frame])) / (np.max(star_frames[frame]) - np.min(star_frames[frame])))
            print(f"\rCooking normalized data: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")

        # Precompute surface
        precompiled_surface = []
        for frame in range(animation_frames):
            precompiled_surface.append(pygame.surfarray.make_surface(np.rot90(np.stack((
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_r_interp),
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_g_interp),
                    np.interp(normalized_data[frame], self.removal_colormap, self.removal_b_interp),
            ), axis=-1)).astype(np.uint8)))
            print(f"\rCooking surface: {frame+1}/{animation_frames}", end="", flush=True)
        print(" Done")
        
        return star_frames, normalized_data, precompiled_surface