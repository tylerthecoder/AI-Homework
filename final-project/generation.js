class Generation {
  constructor() {
    this.pool = new Pool();
  }

  async watch() {
    this.pool.fight();
  }

  async train(draw) {
    const newPool = await this.pool.fight(draw);
    this.pool = new Pool(newPool);
  }

  async trainRep(amount, draw) {
    for (let i = 0; i < amount; i++) {
      await this.train(draw);
    }
  }

  playAgainst() {
    const bestNet = this.pool.getBestPlayer();
    game.playAgainst(bestNet);
  }
}