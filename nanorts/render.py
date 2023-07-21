import pygame
from nanorts.game import Game

class Render:
    def __init__(self, map_height, map_width):
        self.map_height = map_height
        self.map_width = map_width
        self.shape_size = 50
        self.bordersize = self.shape_size // 8
        self.line_width = self.shape_size // 8
        self.vacant = 20
        self.info_height = 100
        
        pygame.init()
        self.viewer = None
        pygame.display.set_caption("RTS Game")
        self.surface = None
        self.clock = pygame.time.Clock()

        self.PLAYER_COLORS = {-1:(0,255,0), 0:(255,0,0), 1:(0,0,255)}
        self.UNIT_TYPE_COLORS = {
            'Base': (128, 128, 128),
            'Worker': (128, 128, 128),
            'Barracks': (64, 64, 64),
            'Light': (255, 255, 0),
            'Heavy': (0, 255, 255),
            'Ranged': (255, 0, 255),
            'Resource': (0, 255, 0),
            'terrain': (0, 128, 0)
        }
        self.RECT_UNITS = {'Base', 'Barracks', 'Resource', 'terrain'}
        self.CIRCLE_UNITS = {'Worker', 'Light', 'Heavy', 'Ranged'}
    
    def draw(self, game:Game):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        viewer_height = game.height * self.shape_size + self.vacant * 2 + self.info_height
        viewer_width = game.width * self.shape_size + self.vacant * 2
        if self.viewer is None:
            self.viewer = pygame.display.set_mode((viewer_width, viewer_height))
        #self.viewer.fill((0,0,0))
        self.surface = pygame.Surface((viewer_width, viewer_height), pygame.SRCALPHA)
        self.surface.fill((0,0,0))
        x_start = self.vacant
        x_end = self.vacant + game.width * self.shape_size
        y_start = self.vacant
        y_end = self.vacant + game.height * self.shape_size
        for i in range(game.width+1):
            x = x_start + i * self.shape_size
            pygame.draw.line(self.surface, (255,255,255), (x, y_start), (x, y_end))
        for i in range(game.height+1):
            y = y_start + i * self.shape_size
            pygame.draw.line(self.surface, (255,255,255), (x_start, y), (x_end, y))
        for unit in game.units.values():
            unit_name = unit.unit_type.name
            unit_player_id = unit.player_id
            unit_int_pos = unit.pos
            unit_pos_x = unit_int_pos % game.width
            unit_pos_y = unit_int_pos // game.width
            x = x_start + unit_pos_x * self.shape_size
            y = y_start + unit_pos_y * self.shape_size    
            player_color = self.PLAYER_COLORS[unit_player_id]
            unit_type_color = self.UNIT_TYPE_COLORS[unit_name]
            if unit_name in self.RECT_UNITS:
                pygame.draw.rect(self.surface, player_color, (x, y, self.shape_size, self.shape_size))
                pygame.draw.rect(self.surface, unit_type_color, (x + self.bordersize, y + self.bordersize, self.shape_size - self.bordersize * 2, self.shape_size - self.bordersize * 2))
            elif unit_name in self.CIRCLE_UNITS:
                pygame.draw.circle(self.surface, player_color, (x + self.shape_size // 2, y + self.shape_size // 2), self.shape_size // 2)
                pygame.draw.circle(self.surface, unit_type_color, (x + self.shape_size // 2, y + self.shape_size // 2), self.shape_size // 2 - self.bordersize)
            else:
                raise Exception("unit_name is not valid")
            # if unit is doing action draw a line to target
            if unit.current_action is not None:
                target_pos_int = unit.current_action_target
                target_x = x_start + (target_pos_int % game.width) * self.shape_size + self.shape_size // 2
                target_y = y_start + (target_pos_int // game.width) * self.shape_size + self.shape_size // 2
                pygame.draw.line(self.surface, (255,255,255), (x + self.shape_size // 2, y + self.shape_size // 2), (target_x, target_y),width=self.line_width)
            # if unit is carrying resource show number of resource on its center
            if unit.carried_resource > 0:
                font = pygame.font.SysFont('Arial', 20)
                text = font.render(str(unit.carried_resource), True, (255,255,255))
                self.surface.blit(text, (x + self.shape_size // 2, y + self.shape_size // 2))
            # if unit is base show number of player's resource on its center
            if unit_name == 'Base':
                font = pygame.font.SysFont('Arial', 20)
                text = font.render(str(game.players[unit.player_id].resource), True, (255,255,255))
                self.surface.blit(text, (x + self.shape_size // 2, y + self.shape_size // 2))
            # if unit current hp is less than max hp show hp bar on it
            if unit.current_hp < unit.unit_type.hp:
                max_hp_bar_width = self.shape_size
                hp_bar_width = self.shape_size * unit.current_hp // unit.unit_type.hp
                pygame.draw.rect(self.surface, (255,0,0), (x, y, max_hp_bar_width, self.line_width))
                pygame.draw.rect(self.surface, (0,255,0), (x, y, hp_bar_width, self.line_width))
            if unit.current_action == "produce":
                target_pos_int = unit.current_action_target
                target_pos_x = target_pos_int % game.width
                target_pos_y = target_pos_int // game.width
                target_x = x_start + target_pos_x * self.shape_size
                target_y = y_start + target_pos_y * self.shape_size
                left_produce_time = unit.execute_current_action_time
                left_time_bar = self.shape_size * left_produce_time // unit.building_unit_type.produceTime
                pygame.draw.rect(self.surface, (0,255,0), (target_x, target_y, left_time_bar, self.line_width))
        font = pygame.font.SysFont('Arial', 20)
        text = font.render("Player0 resource: " + str(game.players[0].resource), True, (255,255,255))
        self.surface.blit(text, (self.vacant, self.vacant + game.height * self.shape_size))
        text = font.render("Player1 resource: " + str(game.players[1].resource), True, (255,255,255))
        self.surface.blit(text, (self.vacant, self.vacant + game.height * self.shape_size + self.info_height // 3))
        text = font.render("Game time: " + str(game.game_time), True, (255,255,255))
        self.surface.blit(text, (self.vacant, self.vacant + game.height * self.shape_size + self.info_height // 3 * 2))
        self.viewer.blit(self.surface, (0,0))
        pygame.display.flip()
        self.clock.tick(0)
