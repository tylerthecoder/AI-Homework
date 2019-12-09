let ctx;
let sw;
let sh;

let drawFight = true;

const PADDLE_SPEED = 15;
const PADDLE_HEIGHT = 130;
const PADDLE_WIDTH = 30;
const PADDLE_OFFSET = 50;
const BALL_RADIUS = 10;
const POOL_SIZE = 2;

const INPUT_LAYER_SIZE = 4;
const HIDDEN_LAYER_SIZE = 8;
const OUTPUT_LAYER_SIZE = 1;

const NET_SIZE =
  INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE;

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
};

// all the vars that change over time
let ballX = 600;
let ballY = 600;
let ballVx = 10;
let ballVy = 10;
let playerA_Y = 0;
let playerB_Y = 0;

const controlWithKeyboard = () => {
  if (keys.has("ArrowUp")) {
    playerA_Y -= PADDLE_SPEED;
  }
  if (keys.has("ArrowDown")) {
    playerA_Y += PADDLE_SPEED;
  }
};

const draw = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

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

const update = () => {
  ballX += ballVx;
  ballY += ballVy;
};

const updateWithAi = (net1, net2) => {
  update();
  playerA_Y += ai(net1, PADDLE_OFFSET + PADDLE_WIDTH, playerA_Y)
    ? PADDLE_SPEED
    : -PADDLE_SPEED;
  playerB_Y += ai(net2, sw - (PADDLE_OFFSET + PADDLE_WIDTH), playerB_Y)
    ? PADDLE_SPEED
    : -PADDLE_SPEED;
};

const constrainValues = () => {
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
    ballVx *= -1.01;
  }

  if (
    ballY > playerB_Y &&
    ballY < playerB_Y + PADDLE_HEIGHT &&
    ballX + BALL_RADIUS > sw - (PADDLE_OFFSET + PADDLE_WIDTH)
  ) {
    ballVx *= -1.01;
  }

  // collision detection for walls
  if (ballY > sh - BALL_RADIUS) {
    ballY = sh - BALL_RADIUS;
    ballVy *= -1;
  }
  if (ballY < BALL_RADIUS) {
    ballY = BALL_RADIUS;
    ballVy *= -1;
  }

  // A lost
  if (ballX < BALL_RADIUS) {
    ballX = BALL_RADIUS;
    ballVx *= -1;
    return -1;
  }
  // B lost
  if (ballX > sw - BALL_RADIUS) {
    ballX = sw - BALL_RADIUS;
    ballVx *= -1;
    return 1;
  }

  return 0;
};

const loop = () => {
  controlWithKeyboard();
  updateWithAi(net1, net2);
  constrainValues();
  draw();
  window.requestAnimationFrame(loop);
};

const pvp = async (net1, net2) => {
  // have two ai's play
  // reset the variables

  ballX = sw / 2;
  ballY = sh / 2;
  playerA_Y = 0;
  playerB_Y = 0;
  ballVx = 5 * (Math.random() > 0.5 ? 1 : -1);
  ballVy = 5 * (Math.random() > 0.5 ? 1 : -1);

  const step = () => {
    count++;
    updateWithAi(net1, net2);
    return constrainValues();
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
        draw();
        if (winner === 0) {
          requestAnimationFrame(stepWithDraw);
        } else {
          resolve(winner);
        }
      };

      stepWithDraw();
    });
  }

  return [winner, count];
};

const genFight = async pool => {
  const newPool = [];
  const p = pool.slice(0);
  while (p.length) {
    const net1 = p.pop();
    const net2 = p.pop();
    const [winner, count] = await pvp(net1, net2);
    console.log(winner, count);
    if (winner === 1) {
      // do mutations as well
      newPool.push(...[net1, net1, net1, net2]);
    } else {
      newPool.push(...[net2, net2, net2, net1]);
    }
  }

  const newPoolReduced = [];
  for (let i = 0; i < POOL_SIZE; i++) {
    const [net1] = newPool.splice(
      Math.floor(Math.random() * newPool.length),
      1
    );
    const [net2] = newPool.splice(
      Math.floor(Math.random() * newPool.length),
      1
    );

    const newNet = mate(net1, net2);

    newPoolReduced.push(newNet);
  }

  return newPoolReduced;
};

const mate = (net1, net2) => {
  // cross over 1. (swap all element after given index)

  const flipIndex = Math.floor(Math.random() * NET_SIZE);

  return net1.slice(0, flipIndex).concat(net2.slice(flipIndex));
};

const generatePool = () => {
  const pool = [];
  for (let i = 0; i < POOL_SIZE; i++) {
    pool.push(generateNet());
  }
  return pool;
};

const generateNet = () => {
  return Array.from(Array(NET_SIZE)).map(() => Math.random() * 2 - 1);
};

const calcNet = (inputs, net) => {
  return Array.from(Array(HIDDEN_LAYER_SIZE))
    .map((_, hiddenLayerIndex) => {
      const value = inputs.reduce((acc, input, inputLayerIndex) => {
        const netIndex = hiddenLayerIndex + inputLayerIndex * HIDDEN_LAYER_SIZE;
        return acc + input * net[netIndex];
      }, 0);
      return value;
    })
    .reduce((acc, val, index) => {
      const netIndex =
        HIDDEN_LAYER_SIZE * INPUT_LAYER_SIZE + index * OUTPUT_LAYER_SIZE;
      return acc + val * net[netIndex];
    });
};

// does ai
const ai = (net, px, py) => {
  const paddleHight = py / sh;
  const deltaBall = py - ballY;

  const inputs = [paddleHight, deltaBall, Math.abs(ballVx), Math.abs(ballVy)];

  const upOrDown = calcNet(inputs, net);

  return upOrDown > 0;
};

window.addEventListener("load", startGame);

let pool = generatePool();
