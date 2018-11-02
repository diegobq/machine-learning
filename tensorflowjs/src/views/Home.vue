<template>
  <div class="train">
    <template v-if="mode == 'train'">
        <h1>Click the "Train Model" button to train the classifier with different images</h1>
    </template>
    <template v-else>
        <h1>Voy a intentar predecir tus movimientos</h1>
    </template>

    <b-button-group>
      <b-button variant="success" :disabled="mode === 'train'" v-on:click="changeOption('train')">Train Model</b-button>
      <b-button variant="success" :disabled="mode === 'test'" v-on:click="changeOption('test')">Test Model</b-button>
    </b-button-group>
    <Camera></Camera>
    <template v-if="mode == 'train'">
        <div>
            <b-container class="bv-example-row">
                <b-row>
                    <b-col offset-md="4" cols="4" >
                        <b-input-group>
                            <b-form-input type="text" id="label_text"></b-form-input>

                            <b-input-group-append>
                                <b-btn variant="outline-secondary" v-on:click="trainModel()">Train Model</b-btn>
                            </b-input-group-append>
                        </b-input-group>
                    </b-col>
                </b-row>
            </b-container>
        </div>

        <div class="text-center">
            <b-row>
                <b-col offset-md="4" cols="4" v-for="example in examples" >
                    <b-button variant="primary" v-on:click="updateLabel(example.label)" >
                        {{ example.label }} <b-badge variant="light">{{ example.trainNumber }}</b-badge>
                    </b-button>
                </b-col>
            </b-row>
        </div>
    </template>
    <template v-else>
        <button v-on:click="getLabel()">{{running ? 'Stop' : 'Start'}}</button>
    </template>

    <h1>{{ detected_e }}</h1>
  </div>
</template>

<script>
// @ is an alias to /src
import Camera from "@/components/Camera.vue";
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

export default {
  name: "Home",
  components: {
    Camera
  },
  data: function(){
      return {
          examples: [],
          classifier: null,
          mobilenet: null,
          class: null,
          detected_e: null,
          mode: 'train',
          running: false,
      }
  },
  mounted: function(){
      this.init();
  },
  methods: {
      async init(){
        // load the load mobilenet and create a KnnClassifier
        this.classifier = knnClassifier.create();
        this.mobilenet = await mobilenetModule.load();
      },
      trainModel(){
        let selectedLabel = document.getElementById("label_text").value;
        const key = selectedLabel.toLowerCase();
        this.class = this.examples.findIndex(function(example) {
          return example.label.toLowerCase() === key;
        });
        if (this.class < 0) {
          this.examples.push({
            label: selectedLabel,
            trainNumber: 0
          });
          this.class = this.examples.length - 1;
        }
        this.examples[this.class].trainNumber++;
        this.addExample();
      },
      addExample(){
        const img= tf.fromPixels(this.$children[0].webcam.webcamElement);
        const logits = this.mobilenet.infer(img, 'conv_preds');
        this.classifier.addExample(logits, parseInt(this.class));
      },
      async getLabel() {
          this.running = !this.running;
          while (this.running) {
            if (this.examples.length > 0) {
              const img = tf.fromPixels(this.$children[0].webcam.webcamElement);
              const logits = this.mobilenet.infer(img, 'conv_preds');
              const pred = await this.classifier.predictClass(logits);
              this.detected_e = this.examples[pred.classIndex].label;
            } else {
              this.detected_e = 'Antes de testear, tenés que entrenar tu clasiffier';
            }
          }
          if(!this.running) this.detected_e = '';
      },
      changeOption(mode){
          this.mode = mode;
          this.detected_e = '';
      },
      updateLabel(label){
        document.getElementById("label_text").value = label;
      }
    }
};
</script>
