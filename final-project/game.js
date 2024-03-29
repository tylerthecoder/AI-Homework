// all the vars that change over time
let ctx;
let sw;
let sh;
let ballX = 600;
let ballY = 600;
let ballVx = 10;
let ballVy = 10;
let playerA_Y = 0;
let playerB_Y = 0;
let playerA_vy = 0;
let playerB_vy = 0;

const PADDLE_SPEED = 15;
const PADDLE_HEIGHT = 130;
const PADDLE_WIDTH = 30;
const PADDLE_OFFSET = 50;
const BALL_RADIUS = 10;

const keys = new Set(); // Keeps track of active key presses
window.addEventListener("keydown", e => keys.add(e.key));
window.addEventListener("keyup", e => keys.delete(e.key));

class Game {
  constructor() {
    const canvas = document.getElementById("canvas");
    const setCanvasDims = () => {
      sw = window.innerWidth;
      sh = window.innerHeight - 130;
      canvas.width = sw;
      canvas.height = sh;
    };
    setCanvasDims();
    window.addEventListener("resize", setCanvasDims);

    ctx = canvas.getContext("2d");
  }

  keyboardControls() {
    if (keys.has("ArrowUp")) {
      playerA_vy = -PADDLE_SPEED;
    } else if (keys.has("ArrowDown")) {
      playerA_vy = PADDLE_SPEED;
    } else {
      playerA_vy = 0;
    }
  }

  draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.beginPath();
    ctx.fillStyle = 'white';
    ctx.rect(PADDLE_OFFSET, playerA_Y, PADDLE_WIDTH, PADDLE_HEIGHT);
    ctx.fill();

    ctx.beginPath();
    ctx.rect(
      sw - (PADDLE_OFFSET + PADDLE_WIDTH),
      playerB_Y,
      PADDLE_WIDTH,
      PADDLE_HEIGHT
    );
    ctx.fill();

    ctx.beginPath();
    ctx.arc(ballX, ballY, BALL_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
  }

  update() {
    ballX += ballVx;
    ballY += ballVy;
    playerA_Y += playerA_vy;
    playerB_Y += playerB_vy;
  }

  updateWithAi(net1, net2) {
    const net1Val = net1.play(PADDLE_OFFSET + PADDLE_WIDTH, playerA_Y);
    const net2Val = net2.play(sw - (PADDLE_OFFSET + PADDLE_WIDTH), playerB_Y);
    playerA_vy = net1Val === -1 ? -PADDLE_SPEED : net1Val === 0 ? 0 : PADDLE_SPEED;
    playerB_vy = net2Val === -1 ? -PADDLE_SPEED : net2Val === 0 ? 0 : PADDLE_SPEED;
  }

  constrainValues(net1, net2) {
    if (playerA_Y < 0) {
      playerA_Y = 0;
    }
    if (playerA_Y > sh - PADDLE_HEIGHT) {
      playerA_Y = sh - PADDLE_HEIGHT;
    }
    if (playerB_Y < 0) {
      playerB_Y = 0;
    }
    if (playerB_Y > sh - PADDLE_HEIGHT) {
      playerB_Y = sh - PADDLE_HEIGHT;
    }

    // collision detection for paddles
    if (
      ballY > playerA_Y &&
      ballY < playerA_Y + PADDLE_HEIGHT &&
      ballX - BALL_RADIUS < PADDLE_OFFSET + PADDLE_WIDTH &&
      ballX + BALL_RADIUS > PADDLE_OFFSET + PADDLE_WIDTH
    ) {
      ballVx *= -1.05;
      // ballVy += (playerA_vy / PADDLE_SPEED) * 5
      ballX = PADDLE_OFFSET + PADDLE_WIDTH + BALL_RADIUS + 1;
      if (net1) {
        net1.score++;
        document.getElementById('leftScore').innerHTML = Number(document.getElementById('leftScore').innerHTML) + 1;
      }
    }

    if (
      ballY > playerB_Y &&
      ballY < playerB_Y + PADDLE_HEIGHT &&
      ballX + BALL_RADIUS > sw - (PADDLE_OFFSET + PADDLE_WIDTH) &&
      ballX - BALL_RADIUS < sw - (PADDLE_OFFSET + PADDLE_WIDTH)
    ) {
      ballVx *= -1.05;
      // ballVy += (playerB_vy / PADDLE_SPEED) * 5
      ballX = sw - (BALL_RADIUS + PADDLE_OFFSET + PADDLE_WIDTH + 1);
      if (net2) {
        net2.score++;
        document.getElementById('rightScore').innerHTML = Number(document.getElementById('rightScore').innerHTML) + 1;
      }
    }

    // collision detection for walls
    if (ballY > sh - BALL_RADIUS) {
      ballY = sh - (BALL_RADIUS + 1);
      ballVy *= -1.01;
    }
    if (ballY < BALL_RADIUS) {
      ballY = BALL_RADIUS + 1;
      ballVy *= -1.01;
    }

    // A lost
    if (ballX < BALL_RADIUS) {
      return -1;
    }
    // B lost
    if (ballX > sw - BALL_RADIUS) {
      return 1;
    }

    return 0;
  }

  playAgainst(net) {
    this.draw();
    this.keyboardControls();
    const netVal = net.play(sw - (PADDLE_OFFSET + PADDLE_WIDTH), playerB_Y);
    playerB_vy = netVal === -1 ? -PADDLE_SPEED : netVal === 0 ? 0 : PADDLE_SPEED;
    this.update();

    const res = this.constrainValues();

    if (res !== 0)
      console.log(res);

    if (res === 0) {
      setTimeout(() => {
        this.playAgainst(net);
      }, 20);
    }
  }

  resetValues() {
    ballX = sw / 2;
    ballY = sh / 2;
    playerA_Y = 0;
    playerB_Y = 0;
    ballVx = 5 * (Math.random() > 0.5 ? Math.random() + .5 : -(Math.random() + .5));
    ballVy = 5 * (Math.random() > 0.5 ? Math.random() + .5 : -(Math.random() + .5));
  }

  async pvp(net1, net2, drawFight) {
    this.resetValues();
    const step = () => {
      count++;
      this.updateWithAi(net1, net2);
      this.update();
      return this.constrainValues(net1, net2);
    };

    let winner = 0;

    let count = 0;
    if (!drawFight) {
      while (winner === 0) {
        winner = step();
      }
    } else {
      winner = await new Promise(resolve => {
        const stepWithDraw = () => {
          const winner = step();
          this.draw();
          if (winner === 0) {
            requestAnimationFrame(stepWithDraw);
          } else {
            resolve(winner);
          }
        };

        stepWithDraw();
      });
    }

    document.getElementById('rightScore').innerHTML = 0;
    document.getElementById('leftScore').innerHTML = 0;
    return [winner, count];
  }
}
