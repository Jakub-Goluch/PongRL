
import pygame as pg

class AIPaddle:
    def __init__(self, screen_rect, ball_rect, difficulty):
        self.difficulty = difficulty
        self.screen_rect = screen_rect
        self.ball_Rect = ball_rect
        self.move_up = False
        self.move_down = False
        self.screen_response_area_rect = self.screen_rect

        num = 60

        if self.difficulty == 'easy6':
            num = 2
        elif self.difficulty == 'easy5':
            num = 3
        elif self.difficulty == 'easy4':
            num = 5
        elif self.difficulty == 'easy3':
            num = 10
        elif self.difficulty == 'easy2':
            num = 20
        elif self.difficulty == 'easy1':
            num = 30
        elif self.difficulty == 'easy0':
            num = 60

        # num = 60
            
        surf = pg.Surface([self.screen_rect.width / num, self.screen_rect.height])
        self.screen_response_area_rect = surf.get_rect()
        
    def update(self, ball_rect, ball, paddle_rect):
        if self.screen_response_area_rect.colliderect(ball_rect):
            if ball_rect.centery < paddle_rect.centery:
                if not ball.moving_away_from_AI:
                    self.move_up = True
            elif ball_rect.centery > paddle_rect.centery:
                if not ball.moving_away_from_AI:
                    self.move_down = True
            
    def reset(self):
        '''reset upon each iteration of update'''
        self.move_up = False
        self.move_down = False
