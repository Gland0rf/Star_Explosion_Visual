import pygame
import sys

class Button:
    def __init__(self, x, y, width, height, text, color=(200, 200, 200), hover_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = pygame.font.Font(None, 36)
        self.is_hovered = False
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.hover_color if self.is_hovered else self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
    def check_click(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)