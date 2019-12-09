let ctx;
let sw;
let sh;

const PADDLE_SPEED = 15;
const PADDLE_HEIGHT = 130;
const PADDLE_WIDTH = 30;
const PADDLE_OFFSET = 50;

const BALL_RADIUS = 10;

const startGame = () => {
  const canvas = document.getElementById("canvas");
  const setCanvasDims = () => {
    sw = window.innerWidth;
    sh = window.innerHeight;
    canvas.width = sw;
    canvas.height = sh;
  };
  setCanvasDims();
  window.addEventListener("resize", setCanvasDims);

  ctx = canvas.getContext("2d");

  loop();
};

let ballX = 600;
let ballY = 600;

let ballVx = 10;
let ballVy = 10;

let playerA_Y = 0;
let playerB_Y = 0;

const draw = () => {
  ctx.beginPath();
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
};

const keys = new Set();
window.addEventListener("keydown", e => keys.add(e.key));
window.addEventListener("keyup", e => keys.delete(e.key));

const loop = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (keys.has("ArrowUp")) {
    playerA_Y -= PADDLE_SPEED;
  }
  if (keys.has("ArrowDown")) {
    playerA_Y += PADDLE_SPEED;
  }

  playerA_Y += ai(net1, sw - (PADDLE_OFFSET + PADDLE_WIDTH), playerB_Y)
    ? PADDLE_SPEED
    : -PADDLE_SPEED;
  playerB_Y += ai(net2, sw - (PADDLE_OFFSET + PADDLE_WIDTH), playerB_Y)
    ? PADDLE_SPEED
    : -PADDLE_SPEED;

  ballX += ballVx;
  ballY += ballVy;

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
    ballX - BALL_RADIUS < PADDLE_OFFSET + PADDLE_WIDTH
  ) {
    ballVx *= -1;
  }

  if (
    ballY > playerB_Y &&
    ballY < playerB_Y + PADDLE_HEIGHT &&
    ballX + BALL_RADIUS > sw - (PADDLE_OFFSET + PADDLE_WIDTH)
  ) {
    ballVx *= -1;
  }

  // collision detection for walls
  if (ballX < BALL_RADIUS) {
    ballX = BALL_RADIUS;
    ballVx *= -1;
  }
  if (ballY > sh - BALL_RADIUS) {
    ballY = sh - BALL_RADIUS;
    ballVy *= -1;
  }

  // death
  if (ballX > sw - BALL_RADIUS) {
    ballX = sw - BALL_RADIUS;
    ballVx *= -1;
  }
  if (ballY < BALL_RADIUS) {
    ballY = BALL_RADIUS;
    ballVy *= -1;
  }

  draw();
  window.requestAnimationFrame(loop);
};

const generateNet = () => {
  // generate the network
  const layer1 = [];
  for (let i = 0; i < 4; i++) {
    const layer = Array.from(Array(8)).map(() => Math.random() * 2 - 1);
    layer1.push(layer);
  }

  const layer2 = [];
  for (let i = 0; i < 8; i++) {
    const node = Array.from(Array(1)).map(() => Math.random() * 2 - 1);
    layer2.push(node);
  }

  return [layer1, layer2];
};

const net1 = generateNet();
const net2 = generateNet();

const calcNet = (inputs, net) => {
  const values = [];

  for (let i = 0; i < 8; i++) {
    const value = inputs.reduce((acc, input, index) => {
      return acc + input * net[0][index][i];
    }, 0);
    values.push(value);
  }

  return values.reduce((acc, val, index) => {
    return acc + val * net[1][index][0];
  });
};

// does ai
const ai = (net, px, py) => {
  // inputs: delta-wall delta-ball ball-v
  // outputs: v

  const deltaWall = Math.min(py, sh - py);
  const deltaBall = Math.abs(px - ballX);

  const inputs = [deltaWall, deltaBall, ballVx, ballVy];

  const upOrDown = calcNet(inputs, net);

  return upOrDown > 0;
};

window.addEventListener("load", startGame);
