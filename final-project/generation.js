class Generation {
  constructor() {
    this.pool = new Pool();
  }

  async train(draw) {
    const newPool = await this.pool.fight(draw);
    console.log(newPool);
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