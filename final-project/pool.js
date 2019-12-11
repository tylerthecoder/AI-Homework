const POOL_SIZE = 20;
class Pool {
  constructor(pop) {
    if (pop) {
      this.pool = pop;
    } else {
      this.pool = [];
      for (let i = 0; i < POOL_SIZE; i++) {
        this.pool.push(new DNA());
      }
    }
  }

  sortPool() {
    this.pool.sort((net1, net2) => {
      return net2.score - net1.score;
    })
  }

  getBestPlayer() {
    this.sortPool();
    return this.pool[0];
  }

  async fight(draw) {
    const newPool = [];
    for (let i = 0; i < POOL_SIZE / 2; i++) {
      const net1 = this.pool[2 * i];
      const net2 = this.pool[2 * i + 1];
      await game.pvp(net1, net2, draw);
    }

    this.sortPool();

    const hightestScore = this.pool[0].score;

    const maxAmountToAdd = 1000;

    // only add the top 90% performers
    for (let i = 0; i < Math.ceil(.9 * POOL_SIZE); i++) {
      const amountToAdd = Math.floor((this.pool[i].score / hightestScore) * maxAmountToAdd);
      const addMembers = Array.from(Array(amountToAdd)).fill(this.pool[i]);
      newPool.push(...addMembers);
    }

    // now that we have the pool mate the rest;

    const matedPool = [this.pool[0], this.pool[1]];

    for (let i = 0; i < POOL_SIZE; i++) {
      const randomIndex1 = Math.floor(Math.random() * newPool.length);
      const [net1] = newPool.splice(randomIndex1, 1);
      const randomIndex2 = Math.floor(Math.random() * newPool.length);
      const [net2] = newPool.splice(randomIndex2, 1);

      console.log(net1.score, net2.score);

      const num = Math.random();
      let newNet;
      if (num > .75) {
        newNet = net1.mate1(net2);
      } else if (num > .5) {
        newNet = net1.mate2(net2);
      } else if (num > .25) {
        newNet = net1;
      } else {
        newNet = net2;
      }

      matedPool.push(newNet);
    }


    return matedPool
  }

}