import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class TicTacToeEnv3x3(gym.Env):
    metadata = {
        "render_modes":
            [
                "human",
                "rgb_array"
            ],
        "render_fps": 4
    }

    def __init__(self, render_mode=None):
        self.window_size = 900  # The size of the PyGame window

        self.board = np.zeros((3, 3), dtype=np.int64)

        # player can be 1 or -1
        self.player = 1

        # We have 9 actions, corresponding to each tile
        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Box(-1, 1, shape=(3, 3), dtype=np.int64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.board

    def _get_info(self):
        return {'player': self.player}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((3, 3), dtype=np.int64)
        self.player = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def check_win(self):
        """
        Функция проверяет состояние на победу
        :return: возвращает 1 если игрок победил, иначе 0
        """
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] != 0:
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] != 0:
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != 0:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != 0:
            return True
        return False

    def check_tie(self):
        return not any(self.board.flatten() == 0)

    def step(self, action):
        x, y = action // 3, action % 3
        if self.board[x][y] != 0:
            observation = self._get_obs()
            info = self._get_info()
            return observation, 0, False, False, info
        self.board[x][y] = self.player
        self.player *= -1

        is_tie = self.check_tie()
        win = self.check_win()
        terminated = win or is_tie

        reward = int(win) if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / 3
        )  # The size of a single grid square in pixels

        # Add gridlines
        for x in range(3 + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        colors = {'X': (0, 0, 255), 'O': (255, 0, 0)}
        pygame.font.init()
        font = pygame.font.Font(None, 256)
        for i in range(3):
            for j in range(3):
                x = j * self.window_size / 3
                y = i * self.window_size / 3
                if self.board[i][j] == 1:
                    symbol = "X"
                elif self.board[i][j] == -1:
                    symbol = "O"
                else:
                    continue
                text = font.render(symbol, True, colors[symbol])
                text_rect = text.get_rect()
                text_rect.center = (x + self.window_size / 6, y + self.window_size / 6)
                canvas.blit(text, text_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
