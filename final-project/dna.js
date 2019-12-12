const INPUT_LAYER_SIZE = 3;
const HIDDEN_LAYER_SIZE = 8;
const OUTPUT_LAYER_SIZE = 3;
const NET_SIZE =
  INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE;

class DNA {
  constructor(genes) {
    this.score = 0;
    if (genes) {
      this.genes = genes;
    } else {
      this.genes = Array.from(Array(NET_SIZE)).map(() => Math.random() * 2 - 1);
    }
  }

  copy() {
    return new DNA(this.genes);
  }

  calculateValue(inputs) {
    return Array.from(Array(HIDDEN_LAYER_SIZE))
      .map((_, hiddenLayerIndex) => {
        const value = inputs.reduce((acc, input, inputLayerIndex) => {
          const netIndex =
            hiddenLayerIndex + inputLayerIndex * HIDDEN_LAYER_SIZE;
          return acc + input * this.genes[netIndex];
        }, 0);
        return value;
      })
      .reduce((acc, val, hiddenLayerIndex) => {
        return acc.map((prevVal, outputLayerIndex) => {
          const netIndex =
            HIDDEN_LAYER_SIZE * INPUT_LAYER_SIZE + hiddenLayerIndex * OUTPUT_LAYER_SIZE + outputLayerIndex;
          return prevVal + this.genes[netIndex] * val;
        })
      }, [0, 0, 0]);
  }

  mate1(net) {
    const flipIndex = Math.floor(Math.random() * NET_SIZE);
    const newGenes = this.genes
      .slice(0, flipIndex)
      .concat(net.genes.slice(flipIndex));
    const newNet = new DNA(newGenes);
    return newNet;
  }

  mate2(net) {
    const newGenes = this.genes.map((gene, index) => {
      if (Math.random() > .3) {
        return net.genes[index];
      }
      return gene;
    })
    const newNet = new DNA(newGenes);
    return newNet;
  }

  mutate() {
    this.genes = this.genes.map(gene => {
      if (Math.random() > .2) {
        return Math.random() * 2 - 1
      }
      return gene;
    })
  }

  play(px, py) {
    const paddleHight = py / sh;
    const deltaBallX = px - ballX;
    const deltaBallY = py - ballY;

    const inputs = [paddleHight, deltaBallY, deltaBallX];

    const outputs = this.calculateValue(inputs);

    let largestIndex = 0;
    let largestVal = outputs[0];
    for (let i = 1; i < OUTPUT_LAYER_SIZE; i++) {
      if (largestVal > outputs[i]) {
        largestVal = outputs[i];
        largestIndex = i;
      }
    }

    return largestIndex - 1;
  }
}

