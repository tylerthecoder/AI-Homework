class Generation {
  constructor() {
    this.pool = new Pool();
    this.pastGenerations = [];
  }

  async watch() {
    this.pool.fight();
  }

  async train(draw) {
    const newPool = await this.pool.fight(draw);
    this.pastGenerations.push(this.pool);
    this.pool = new Pool(newPool);
  }

  async trainRep(amount, draw) {
    for (let i = 0; i < amount; i++) {
      await this.train(draw);
    }
  }

  playAgainst() {
    game.resetValues();
    const lastGen = this.pastGenerations[this.pastGenerations.length - 1];
    if (!lastGen) {
      alert("No trained generations");
      return true;
    }
    const bestNet = lastGen.getBestPlayer();
    console.log("Playing against", bestNet);
    game.playAgainst(bestNet);
  }
}