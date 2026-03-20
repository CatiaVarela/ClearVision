import pygame
import time

class AudioManager:

    def __init__(self):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("sounds/alert.wav")
        self.last_alert = 0
        self.cooldown = 2

    def play_alert(self, alert):
        now = time.time()

        if now - self.last_alert < self.cooldown:
            return

        self.last_alert = now

        print(f"🔊 ALERTE: {alert}")
        self.sound.play()