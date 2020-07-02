/** Eigen3 backend.
 *
 * Only supports float32 computation with NHWC memory layout (at runtime and as input).
 * Does not currently support symmetries.
 */

// CR lpuchallafiore: Add support for symmetries.
// CR lpuchallafiore: Add multi-threading support (see "Evaluating with a Thread Pool" in the Eigen Tensor docs).

#include "../neuralnet/nninterface.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <zstr/src/zstr.hpp>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"

#define SCALAR float

using namespace std;

using Eigen::Tensor;

// Debugging -----------------------------------------------------------------------------------------------------------
// #define DEBUG true

template <typename T>
void printTensorShape(const string& name, const T& t) {
  auto d = t.dimensions();
  cout << name << " rank=" << d.size() << " - (";
  for (int i = 0; i < d.size(); i++) {
    cout << d[i] << ",";
  }
  cout << ")" << endl;
}

#if DEBUG
#define DSHAPE(n, x) printTensorShape(n,x)
#define DTENSOR(n, x) cout << n << x << endl
#else
#define DSHAPE(n, x)
#define DTENSOR(n, x)
#endif

// NHWC
void printTensor4(const string& name, const Tensor<SCALAR, 4>& t) {
  printTensorShape(name, t);
  for(int n = 0; n < t.dimension(3); n++) {
    cout << "n = " << n << endl;
    for(int h = 0; h < t.dimension(2); h++) {
      for(int w = 0; w < t.dimension(1); w++) {
        for(int c = 0; c < t.dimension(0); c++) {
          cout << t(c, w, h, n) << (c == t.dimension(0) - 1 ? ", " : " ");
        }
      }
      cout << endl;
    }
    cout << endl;
  }
}

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(istream& in) { modelDesc = std::move(ModelDesc(in)); }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

// Layers --------------------------------------------------------------------------------------------------------------

// Convolution layer with zero-padding.
struct ConvLayer {
  string name;

  Eigen::array<pair<int, int>, 4> paddings;
  Tensor<SCALAR, 4> kernel;
  int inChannels, outChannels;
  int paddingX, paddingY;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc) {
    name = desc.name;
    int convYSize = desc.convYSize;
    int convXSize = desc.convXSize;
    inChannels = desc.inChannels;
    outChannels = desc.outChannels;
    // CR lpuchallafiore: dilation?
    int dilationY = desc.dilationY;
    int dilationX = desc.dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    paddings[0] = make_pair(0, 0);                // C
    paddings[1] = make_pair(paddingX, paddingX);  // W
    paddings[2] = make_pair(paddingY, paddingY);  // H
    paddings[3] = make_pair(0, 0);                // N

    // CR-someday lpuchallafiore: optimize NHWC vs NCHW, etc.
    kernel = Eigen::TensorMap<const Tensor<SCALAR, 4>>(
      (float*)&desc.weights[0], convXSize, convYSize, inChannels, outChannels);
  }

  void apply(const Tensor<SCALAR, 4>& input, Tensor<SCALAR, 4>* output, bool accumulate) const {
    auto padded = input.pad(paddings);

    auto out = Tensor<SCALAR, 4>(outChannels, input.dimension(1), input.dimension(2), input.dimension(3));
    for(int n = 0; n < input.dimension(3); n++) {
      auto inN = padded.chip(n, 3);
      for(int oc = 0; oc < outChannels; oc++) {
        Tensor<SCALAR, 2> sum(input.dimension(1), input.dimension(2));
        sum.setZero();

        for(int ic = 0; ic < inChannels; ic++) {
          Eigen::array<ptrdiff_t, 2> dims({0, 1});
          auto kChip = kernel.chip(oc, 3).chip(ic, 2);
          auto inNC = inN.chip(ic, 0);
          sum += inNC.convolve(kChip, dims);
        }

        out.chip(n, 3).chip(oc, 0) = sum;
      }
    }

    if (accumulate) {
      *output = *output + out;
    } else {
      *output = out;
    }
  }
};

struct BatchNormLayer {
  string name;
  int numChannels;
  float epsilon;
  int xSize;
  int ySize;

  vector<float> mergedScale;
  vector<float> mergedBias;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc& desc) {
    name = desc.name;
    numChannels = desc.numChannels;
    epsilon = desc.epsilon;

    mergedScale.resize(numChannels);
    mergedBias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + epsilon);
      mergedBias[c] = desc.bias[c] - mergedScale[c] * desc.mean[c];
    }
  }

  // Mask should be in 'NHW' format (no "C" channel).
  void apply(bool applyRelu, const Tensor<SCALAR, 4>& input, const Tensor<SCALAR, 3>& mask, Tensor<SCALAR, 4>* output)
    const {
    *output = Tensor<SCALAR, 4>(input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
    for(int c = 0; c < input.dimension(0); c++) {
      auto inC = input.chip(c, 0);
      auto x = inC * mergedScale[c] + mergedBias[c];
      auto z = Tensor<SCALAR, 3>(mask.dimension(0), mask.dimension(1), mask.dimension(2)).setZero();
      if(applyRelu) {
        output->chip(c, 0) = (mask == 1.f).select(x.cwiseMax(0.f), z);
      } else {
        output->chip(c, 0) = (mask == 1.f).select(x, z);
      }
    }
  }
};

struct ActivationLayer {
  string name;

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;

  ActivationLayer(const ActivationLayerDesc& desc) { name = desc.name; }

  template <int N>
  void apply(const Tensor<SCALAR, N>& input, Tensor<SCALAR, N>* output) const { *output = input.cwiseMax(0.f); }
};

struct MatMulLayer {
  string name;
  int inChannels;
  int outChannels;
  Tensor<SCALAR, 2> weights;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(const MatMulLayerDesc& desc)
    : name(desc.name),
      inChannels(desc.inChannels),
      outChannels(desc.outChannels) {
    weights = Tensor<SCALAR, 2>(outChannels, inChannels);
    memcpy(weights.data(), &desc.weights[0], sizeof(SCALAR) * weights.size());
  }

  void apply(const Tensor<SCALAR, 2>& in, Tensor<SCALAR, 2>* out) const {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    *out = weights.contract(in, product_dims);
  }
};

struct MatBiasLayer {
  string name;
  int numChannels;
  std::vector<float> weights;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(const MatBiasLayerDesc& desc)
    : name(desc.name),
      numChannels(desc.numChannels),
      weights(desc.weights) {}

  void apply(Tensor<SCALAR, 2>* mat) const {
    for(int n = 0; n < mat->dimension(1); n++) {
      for(int c = 0; c < mat->dimension(0); c++) {
        (*mat)(c, n) += weights[c];
      }
    }
  }
};

// Blocks
// --------------------------------------------------------------------------------------------------------------

struct ResidualBlockIntf {
  virtual ~ResidualBlockIntf(){}

  virtual void apply(Tensor<SCALAR, 4>* trunk, const Tensor<SCALAR, 3>& mask) = 0;
};

struct ResidualBlock : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(const ResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN),
      preActivation(desc.preActivation),
      regularConv(desc.regularConv),
      midBN(desc.midBN),
      midActivation(desc.midActivation),
      finalConv(desc.finalConv) {}

  Tensor<SCALAR, 4> midIn, trunkScratch, midScratch;

  virtual void apply(Tensor<SCALAR, 4>* trunk, const Tensor<SCALAR, 3>& mask) {
    apply(trunk, mask, &trunkScratch, &midIn, &midScratch);
  }

  void apply(
    Tensor<SCALAR, 4>* trunk,
    const Tensor<SCALAR, 3>& mask,
    Tensor<SCALAR, 4>* trunkScratch,
    Tensor<SCALAR, 4>* midIn,
    Tensor<SCALAR, 4>* midScratch) const {
    const bool applyBNRelu = true;
    preBN.apply(applyBNRelu, *trunk, mask, trunkScratch);
    regularConv.apply(*trunkScratch, midIn, false);
    midBN.apply(applyBNRelu, *midIn, mask, midScratch);
    finalConv.apply(*midScratch, trunk, true);
  }
};

// Given two tensors with shapes inA: [n, h, w, cA] and inB: [n, h, w, cB]
// Copy them into a single tensor out: [n, h, w, cA + cB]
Tensor<SCALAR, 4> concatTensors(const Tensor<SCALAR, 4>& a, const Tensor<SCALAR, 4>& b) {
  assert(a.dimension(1) == b.dimension(1) && a.dimension(2) == b.dimension(2) && a.dimension(3) == b.dimension(3));
  Tensor<SCALAR, 4> x = Tensor<SCALAR, 4>(/* C */ a.dimension(0) + b.dimension(0),
                                          /* W */ a.dimension(1),
                                          /* H */ a.dimension(2),
                                          /* N */ a.dimension(3));
  for (int n = 0; n < a.dimension(3); n++) {
    for (int h = 0; h < a.dimension(2); h++) {
      for (int w = 0; w < a.dimension(1); w++) {
        int c = 0;
        for (int ca = 0; a.dimension(0); ca++, c++) {
          x(c,w,h,n) = a(ca,w,h,n);
        }
        for (int cb = 0; b.dimension(0); cb++, c++) {
          x(c,w,h,n) = b(cb,w,h,n);
        }
      }
    }
  }
  return x;
}

struct DilatedResidualBlock : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer dilatedConv;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  DilatedResidualBlock() = delete;
  DilatedResidualBlock(const DilatedResidualBlock&) = delete;
  DilatedResidualBlock& operator=(const DilatedResidualBlock&) = delete;

  DilatedResidualBlock(const DilatedResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN),
      preActivation(desc.preActivation),
      regularConv(desc.regularConv),
      dilatedConv(desc.dilatedConv),
      midBN(desc.midBN),
      midActivation(desc.midActivation),
      finalConv(desc.finalConv) {}

  Tensor<SCALAR, 4> trunkScratch, midScratch, regularOut, dilatedOut;

  virtual void apply(Tensor<SCALAR, 4>* trunk, const Tensor<SCALAR, 3>& mask) {
    apply(trunk, mask, &trunkScratch, &regularOut, &dilatedOut, &midScratch);
  }

  void apply(
    Tensor<SCALAR, 4>* trunk,
    const Tensor<SCALAR, 3>& mask,
    Tensor<SCALAR, 4>* trunkScratch,
    Tensor<SCALAR, 4>* regularOut,
    Tensor<SCALAR, 4>* dilatedOut,
    Tensor<SCALAR, 4>* midScratch) const {
    const bool applyBNRelu = true;
    preBN.apply(applyBNRelu, *trunk, mask, trunkScratch);
    regularConv.apply(*trunkScratch, regularOut, false);
    dilatedConv.apply(*trunkScratch, dilatedOut, false);
    auto midIn = concatTensors(*regularOut, *dilatedOut);
    midBN.apply(applyBNRelu, midIn, mask, midScratch);
    finalConv.apply(*midScratch, trunk, true);
  }
};

// in NxHxWxC, bias NxC
void addNCBiasInplace(Tensor<SCALAR, 4>* in, const Tensor<SCALAR, 2>& bias) {
  assert(in->dimension(0) == bias.dimension(0) && in->dimension(3) == bias.dimension(1));
  for (int n = 0; n < in->dimension(3); n++) {
    for (int h = 0; h < in->dimension(2); h++) {
      for (int w = 0; w < in->dimension(1); w++) {
        for (int c = 0; c < in->dimension(0); c++) {
          (*in)(c,w,h,n) += bias(c,n);
        }
      }
    }
  }
}

//
void poolRowsGPool(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>* out, const float* maskSum) {
  *out = Tensor<SCALAR, 2>(3 * in.dimension(0), in.dimension(3));

  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      float m = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
          m = max(m, x);
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      (*out)(c, n) = mean;
      (*out)(c + in.dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      (*out)(c + 2*in.dimension(0), n) = m;
    }
  }
}

// Given input [n,w,h,c] fills output of shape [n,c*2] with sum over c and max positive over c, in that order.
void poolRowsSumAndMaxPositive(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>* out, float scaleSum) {
  *out = Tensor<SCALAR, 2>(2 * in.dimension(0), in.dimension(3));

  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      float m = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
          m = max(m, x);
        }
      }
      (*out)(c, n) = s * scaleSum;
      (*out)(c + in.dimension(0), n) = m;
    }
  }
}

// Given input [n,w,h,c] fills output of shape [n,c] with sum over c.
void poolRowsSum(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>* out, float scaleSum) {
  *out = Tensor<SCALAR, 2>(in.dimension(0), in.dimension(3));
  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
        }
      }
      (*out)(c, n) = s * scaleSum;
    }
  }
}

void poolRowsValueHead(const Tensor<SCALAR, 4>& in, Tensor<SCALAR, 2>* out, const float* maskSum) {
  *out = Tensor<SCALAR, 2>(3 * in.dimension(0), in.dimension(3));

  for (int n = 0; n < in.dimension(3); n++) {
    for (int c = 0; c < in.dimension(0); c++) {
      float s = 0.f;
      for (int h = 0; h < in.dimension(2); h++) {
        for (int w = 0; w < in.dimension(1); w++) {
          float x = in(c, w, h, n);
          s += x;
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      (*out)(c, n) = mean;
      (*out)(c + in.dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      (*out)(c + 2*in.dimension(0), n) = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
    }
  }
}

struct GlobalPoolingResidualBlock : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BatchNormLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN),
      preActivation(desc.preActivation),
      regularConv(desc.regularConv),
      gpoolConv(desc.gpoolConv),
      gpoolBN(desc.gpoolBN),
      gpoolActivation(desc.gpoolActivation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      midBN(desc.midBN),
      midActivation(desc.midActivation),
      finalConv(desc.finalConv) {}

  Tensor<SCALAR, 4> trunkScratch, regularOut, gpoolOut, gpoolOut2, regularScratch;
  Tensor<SCALAR, 2> gpoolConcat, gpoolBias;

  virtual void apply(Tensor<SCALAR, 4>* trunk, const Tensor<SCALAR, 3>& mask) {
    int batchSize = trunk->dimension(3);
    int nnYLen = trunk->dimension(2);
    int nnXLen = trunk->dimension(1);
    std::vector<float> maskSum(batchSize);

    for (int n = 0; n < batchSize; n++) {
      float s = 0.f;
      for (int h = 0; h < nnYLen; h++) {
        for (int w = 0; w < nnXLen; w++) {
          s += mask(w, h, n);
        }
      }
      maskSum[n] = s;
    }

    apply(
      trunk,
      &trunkScratch,
      &regularOut,
      &gpoolOut,
      &gpoolOut2,
      &gpoolConcat,
      &gpoolBias,
      &regularScratch,
      mask,
      &maskSum[0]);
  }

  void apply(
    Tensor<SCALAR, 4>* trunk,
    Tensor<SCALAR, 4>* trunkScratch,
    Tensor<SCALAR, 4>* regularOut,
    Tensor<SCALAR, 4>* gpoolOut,
    Tensor<SCALAR, 4>* gpoolOut2,
    Tensor<SCALAR, 2>* gpoolConcat,
    Tensor<SCALAR, 2>* gpoolBias,
    Tensor<SCALAR, 4>* regularScratch,
    const Tensor<SCALAR, 3>& mask,
    const float* maskSum) const {
    const bool applyBNRelu = true;
    int xSize = trunk->dimension(2);
    int ySize = trunk->dimension(1);

    DTENSOR("trunk", *trunk);
    DTENSOR("mask", *mask);
    preBN.apply(applyBNRelu, *trunk, mask, trunkScratch);
    DTENSOR("trunkScratch", *trunkScratch);
    regularConv.apply(*trunkScratch, regularOut, false);
    DTENSOR("regularOut", *regularOut);
    gpoolConv.apply(*trunkScratch, gpoolOut, false);
    DTENSOR("gpoolOut", *gpoolOut);
    gpoolBN.apply(applyBNRelu, *gpoolOut, mask, gpoolOut2);
    DTENSOR("gpoolOut2", *gpoolOut2);
    if (maskSum != nullptr) {
      poolRowsGPool(*gpoolOut2, gpoolConcat, maskSum);
    } else {
      poolRowsSumAndMaxPositive(*gpoolOut2, gpoolConcat, 1.f / (xSize * ySize));
    }
    gpoolToBiasMul.apply(*gpoolConcat, gpoolBias);
    addNCBiasInplace(regularOut, *gpoolBias);
    midBN.apply(applyBNRelu, *regularOut, mask, regularScratch);
    finalConv.apply(*regularScratch, trunk, true);
    DSHAPE("trunk", *trunk);
    DSHAPE("trunkScratch", *trunkScratch);
    DSHAPE("regularOut", *regularOut);
    DSHAPE("gpoolOut", *gpoolOut);
    DSHAPE("gpoolOut2", *gpoolOut2);
    DSHAPE("gpoolConcat", *gpoolConcat);
    DSHAPE("gpoolBias", *gpoolBias);
    DSHAPE("mask", mask);
  }
};

struct Trunk {
  string name;
  int version;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;
  int regularNumChannels;
  int dilatedNumChannels;
  int gpoolNumChannels;

  ConvLayer initialConv;
  MatMulLayer initialMatMul;
  vector<pair<int, ResidualBlockIntf*>> blocks;
  BatchNormLayer trunkTipBN;
  ActivationLayer trunkTipActivation;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(const TrunkDesc& desc)
    : name(desc.name),
      version(desc.version),
      numBlocks(desc.numBlocks),
      trunkNumChannels(desc.trunkNumChannels),
      midNumChannels(desc.midNumChannels),
      regularNumChannels(desc.regularNumChannels),
      dilatedNumChannels(desc.dilatedNumChannels),
      gpoolNumChannels(desc.gpoolNumChannels),
      initialConv(desc.initialConv),
      initialMatMul(desc.initialMatMul),
      trunkTipBN(desc.trunkTipBN),
      trunkTipActivation(desc.trunkTipActivation) {
    for (int i = 0; i < numBlocks; ++i) {
      auto blockDesc = desc.blocks[i];
      if (blockDesc.first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)desc.blocks[i].second;
        ResidualBlockIntf* block = new ResidualBlock(*blockDesc);
        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND, block));
      } else if (blockDesc.first == DILATED_BLOCK_KIND) {
        DilatedResidualBlockDesc* blockDesc = (DilatedResidualBlockDesc*)desc.blocks[i].second;
        ResidualBlockIntf* block = new DilatedResidualBlock(*blockDesc);
        blocks.push_back(make_pair(DILATED_BLOCK_KIND, block));
      } else if (blockDesc.first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)desc.blocks[i].second;
        GlobalPoolingResidualBlock* block = new GlobalPoolingResidualBlock(*blockDesc);
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND, block));
      } else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  virtual ~Trunk() {
    for (auto p : blocks) {
      delete p.second;
    }
  }

  // TODO cleanup a lot of these Tensor inputs (make local vars, etc)
  void apply(
    int batchSize,
    Tensor<SCALAR, 4>* input,
    const Tensor<SCALAR, 2>& inputGlobal,
    const Tensor<SCALAR, 3>& mask,
    Tensor<SCALAR, 4>* trunk,
    Tensor<SCALAR, 4>* trunkScratch,
    Tensor<SCALAR, 4>* regularOut,
    Tensor<SCALAR, 4>* dilatedOut,
    Tensor<SCALAR, 4>* midIn,
    Tensor<SCALAR, 4>* midScratch,
    Tensor<SCALAR, 4>* gpoolOut,
    Tensor<SCALAR, 4>* gpoolOut2,
    Tensor<SCALAR, 2>* gpoolConcat,
    Tensor<SCALAR, 2>* gpoolBias,
    Tensor<SCALAR, 4>* regularScratch) const {

    initialConv.apply(*input, trunkScratch, false);
    Tensor<SCALAR, 2> matMulOut;
    initialMatMul.apply(inputGlobal, &matMulOut);
    addNCBiasInplace(trunkScratch, matMulOut);

    // apply blocks
    // Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
    for (auto block : blocks) {
      block.second->apply(trunkScratch, mask);
    }

    // And now with the final BN port it from trunkScratchBuf to trunkBuf.
    const bool applyBNRelu = true;
    trunkTipBN.apply(applyBNRelu, *trunkScratch, mask, trunk);
  }
};

struct PolicyHead {
  string name;
  int version;
  int xSize;
  int ySize;
  int p1Channels;
  int g1Channels;
  int p2Channels;

  ConvLayer p1Conv;
  ConvLayer g1Conv;
  BatchNormLayer g1BN;
  ActivationLayer g1Activation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer p1BN;
  ActivationLayer p1Activation;
  ConvLayer p2Conv;
  MatMulLayer gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(const PolicyHeadDesc& desc)
    : name(desc.name),
      version(desc.version),
      xSize(desc.p1Conv.convXSize),
      ySize(desc.p1Conv.convYSize),
      p1Channels(desc.p1Conv.outChannels),
      g1Channels(desc.g1Conv.outChannels),
      p2Channels(desc.p2Conv.outChannels),
      p1Conv(desc.p1Conv),
      g1Conv(desc.g1Conv),
      g1BN(desc.g1BN),
      g1Activation(desc.g1Activation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      p1BN(desc.p1BN),
      p1Activation(desc.p1Activation),
      p2Conv(desc.p2Conv),
      gpoolToPassMul(desc.gpoolToPassMul) {}

  void apply(Tensor<SCALAR, 3>* mask,
             float* maskSum,
             Tensor<SCALAR, 4>* trunk,
             Tensor<SCALAR, 4>* p1Out,
             Tensor<SCALAR, 4>* p1Out2,
             Tensor<SCALAR, 4>* g1Out,
             Tensor<SCALAR, 4>* g1Out2,
             Tensor<SCALAR, 2>* g1Concat,
             Tensor<SCALAR, 2>* g1Bias,
             Tensor<SCALAR, 4>* p2Out,
             Tensor<SCALAR, 2>* g1Pass,
             Tensor<SCALAR, 4>* policy) const {
    const bool applyBNRelu = true;
    p1Conv.apply(*trunk, p1Out, false);
    g1Conv.apply(*trunk, g1Out, false);
    g1BN.apply(applyBNRelu, *g1Out, *mask, g1Out2);
    const float meanScale = 1.0f / (xSize * ySize);
    if (maskSum != nullptr) {
      poolRowsGPool(*g1Out2, g1Concat, maskSum);
    } else {
      poolRowsSumAndMaxPositive(*g1Out2, g1Concat, meanScale);
    }
    gpoolToBiasMul.apply(*g1Concat, g1Bias);
    Tensor<SCALAR, 4>* p1OutA = p1Out;
    Tensor<SCALAR, 4>* p1OutB = p1Out2;
    addNCBiasInplace(p1OutA, *g1Bias);
    p1BN.apply(true, *p1OutA, *mask, p1OutB);
    p2Conv.apply(*p1OutB, p2Out, false);
    // TODO-someday: apply symmetries
    gpoolToPassMul.apply(*g1Concat, g1Pass);
    *policy = concatTensors(*p2Out, *g1Pass); // wxh -> 1?
  }
};

struct ValueHead {
  string name;
  int version;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  int scoreValueChannels;
  int ownershipChannels;
  int xSize, ySize;

  ConvLayer v1Conv;
  BatchNormLayer v1BN;
  ActivationLayer v1Activation;
  MatMulLayer v2Mul;
  MatBiasLayer v2Bias;
  ActivationLayer v2Activation;
  MatMulLayer v3Mul;
  MatBiasLayer v3Bias;
  MatMulLayer sv3Mul;
  MatBiasLayer sv3Bias;
  ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(const ValueHeadDesc& desc)
    : name(desc.name),
      version(desc.version),
      xSize(desc.v1Conv.convXSize),
      ySize(desc.v1Conv.convYSize),
      v1Channels(desc.v1Conv.outChannels),
      v2Channels(desc.v2Mul.outChannels),
      valueChannels(desc.sv3Mul.outChannels),
      scoreValueChannels(desc.vOwnershipConv.outChannels),
      v1Conv(desc.v1Conv),
      v1BN(desc.v1BN),
      v1Activation(desc.v1Activation),
      v2Mul(desc.v2Mul),
      v2Bias(desc.v2Bias),
      v2Activation(desc.v2Activation),
      v3Mul(desc.v3Mul),
      v3Bias(desc.v3Bias),
      sv3Mul(desc.sv3Mul),
      sv3Bias(desc.sv3Bias),
      vOwnershipConv(desc.vOwnershipConv) {}

  void apply(Tensor<SCALAR, 4>* mask,
             float* maskSum,
             Tensor<SCALAR, 4>* trunk,
             Tensor<SCALAR, 4>* v1Out,
             Tensor<SCALAR, 4>* v1Out2,
             Tensor<SCALAR, 2>* v1Mean,
             Tensor<SCALAR, 2>* v2Out,
             Tensor<SCALAR, 2>* value,
             Tensor<SCALAR, 2>* scoreValue,
             Tensor<SCALAR, 4>* ownership) const {
    bool applyBNRelu = true;
    v1Conv.apply(*trunk, v1Out, false);
    v1BN.apply(applyBNRelu, *v1Out, *mask, v1Out2);
    const float meanScale = 1.0f / (xSize * ySize);
    if (maskSum != nullptr) {
      poolRowsValueHead(*v1Out2, v1Mean, maskSum);
    } else {
      poolRowsSum(*v1Out2, v1Mean, meanScale);
    }
    v2Mul.apply(*v1Mean, v2Out);
    v2Bias.apply(v2Out);
    v2Activation.apply(*v2Out, v2Out); // 4 -> 2?
    v3Mul.apply(*v2Out, value);
    v3Bias.apply(value);

    sv3Mul.apply(*v2Out, scoreValue);
    sv3Bias.apply(scoreValue);

    vOwnershipConv.apply(*v1Out2, ownership, false);
    // TODO-someday: apply symmetries
  }
};

// Model and Buffer I/O ------------------------------------------------------------------------------------------------

struct Model {
  string name;
  int version;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  Trunk trunk;
  PolicyHead policyHead;
  ValueHead valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(const ModelDesc& desc)
    : name(desc.name), version(desc.version), numInputChannels(desc.numInputChannels),
      numInputGlobalChannels(desc.numInputGlobalChannels),
      numValueChannels(desc.numValueChannels),
      numScoreValueChannels(desc.numScoreValueChannels),
      numOwnershipChannels(desc.numOwnershipChannels),
      trunk(desc.trunk),
      policyHead(desc.policyHead),
      valueHead(desc.valueHead) {}

  void apply(void* input,
             void* inputGlobal) const {
    // TODO: fill mask
    // TODO: apply Trunk
    // TODO: apply PolicyHead
    // TODO: apply ValueHead
  }
};

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;

  // Eigen tensors are stored in column-major order, so an NHWC memory layout is given by Tensor<4>(C,W,H,N).
  Tensor<SCALAR, 4> spatialInput;
  Tensor<SCALAR, 4> globalInput;
  bool* symmetriesBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = m.version >= 3 ? nnXLen : m.xSizePreV3;
    int ySize = m.version >= 3 ? nnYLen : m.ySizePreV3;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * xSize * ySize;
    singleInputGlobalElts = m.numInputGlobalChannels;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    spatialInput = Tensor<SCALAR, 4>(m.numInputChannels, xSize, ySize, maxBatchSize);
    globalInput = Tensor<SCALAR, 4>(1, 1, m.numInputGlobalChannels, maxBatchSize);

    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];
  }

  ~InputBuffers() { delete[] symmetriesBuffer; }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int rowIdx) {
  assert(rowIdx < inputBuffers->maxBatchSize);
  return inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * rowIdx);
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}
int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputGlobalElts;
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;

  try {
    // zstr has a bad property of simply aborting if the file doesn't exist
    // So we try to catch this common error by explicitly testing first if the
    // file exists by trying to open it normally to turn it into a regular C++
    // exception.
    {
      ifstream testIn(file);
      if(!testIn.good())
        throw StringError("File does not exist or could not be opened");
    }
    zstr::ifstream in(file);
    LoadedModel* loadedModel = new LoadedModel(in);
    return loadedModel;
  } catch(const StringError& e) {
    throw StringError("Error parsing model file " + file + ": " + e.what());
  }
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

// NeuralNet -----------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // no-op for cpu
}

void NeuralNet::globalCleanup() {
  // no-op for cpu
}

struct ComputeHandle {
  // unique_ptr<Model> model;
  int nnXLen;
  int nnYLen;
  bool requireExactPosLen;
  int policySize;

  ComputeHandle(const LoadedModel& loadedModel, int maxBatchSize, int xLen, int yLen, bool rExactPosLen)
    :  // model(make_unique<Model>(loadedModel.modelDesc, maxBatchSize, xLen, yLen)),
      nnXLen(xLen),
      nnYLen(yLen),
      requireExactPosLen(rExactPosLen),
      policySize(NNPos::getPolicySize(nnXLen, nnYLen)) {}
};

ComputeHandle* NeuralNet::createComputeHandle(
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool requireExactPosLen,
  bool inputsUseNHWC,
  int cudaGpuIdxForThisThread,
  bool useFP16,
  bool cudaUseNHWC) {
  (void)cudaUseNHWC;      // Always use NHWC
  (void)useFP16;          // Always use FP32
  assert(inputsUseNHWC);  // Only support inputs in NHWC format.
  return new ComputeHandle(*loadedModel, maxBatchSize, nnXLen, nnYLen, requireExactPosLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* buffers,
  int numFilledRows,
  vector<NNOutput*>& outputs) {
  assert(false);
}

// FOR TESTING ---------------------------------------------------------------------------------------------------------
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    ConvLayer layer(*desc);
    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->inChannels, nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> outTensor;

    layer.apply(inTensor, &outTensor, false);

    outputBuffer.resize(outTensor.size());
    memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
    return true;
  } else {
    return false;
  }
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    BatchNormLayer layer(*desc);
    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->numChannels, nnXLen, nnYLen, batchSize);
    Eigen::TensorMap<const Tensor<SCALAR, 3>> maskTensor((float*)&maskBuffer[0], nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> outTensor;

    layer.apply(false, inTensor, maskTensor, &outTensor);

    outputBuffer.resize(outTensor.size());
    memcpy(&outputBuffer[0], outTensor.data(), sizeof(SCALAR) * outTensor.size());
    return true;
  } else {
    return false;
  }
}

// CR lpuchallafiore: test accumulate=true conv layer.
// CR lpuchallafiore: test evaluate activation layer.
// CR lpuchallafiore: test evaluate matmul layer.
// CR lpuchallafiore: test evaluate matbias layer.

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    ResidualBlock block(*desc);
    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
    Eigen::TensorMap<const Tensor<SCALAR, 3>> maskTensor((float*)&maskBuffer[0], nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> midIn(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> trunkScratch, midScratch;
    Tensor<SCALAR, 4> trunk = inTensor;

    block.apply(&trunk, maskTensor, &trunkScratch, &midIn, &midScratch);

    outputBuffer.resize(trunk.size());
    memcpy(&outputBuffer[0], trunk.data(), sizeof(SCALAR) * trunk.size());

    return true;
  } else {
    return false;
  }
}

// CR lpuchallafiore: test evaluate dilatedresidualblock

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  if(useNHWC && !useFP16) {
    GlobalPoolingResidualBlock block(*desc);

    Eigen::TensorMap<const Tensor<SCALAR, 4>> inTensor(
      (float*)&inputBuffer[0], desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
    Tensor<SCALAR, 4> trunk = inTensor;
    Tensor<SCALAR, 4> trunkScratch, regularOut, gpoolOut, gpoolOut2, regularScratch;
    Tensor<SCALAR, 2> gpoolConcat, gpoolBias;
    Eigen::TensorMap<const Tensor<SCALAR, 3>> maskTensor((float*)&maskBuffer[0], nnXLen, nnYLen, batchSize);
    std::vector<float> maskSum(batchSize);

    for (int n = 0; n < batchSize; n++) {
      float s = 0.f;
      for (int h = 0; h < nnYLen; h++) {
        for (int w = 0; w < nnXLen; w++) {
          s += maskTensor(w, h, n);
        }
      }
      maskSum[n] = s;
    }

    block.apply(
      &trunk,
      &trunkScratch,
      &regularOut,
      &gpoolOut,
      &gpoolOut2,
      &gpoolConcat,
      &gpoolBias,
      &regularScratch,
      maskTensor,
      &maskSum[0]);

    outputBuffer.resize(trunk.size());
    memcpy(&outputBuffer[0], trunk.data(), sizeof(SCALAR) * trunk.size());

    return true;
  } else {
    return false;
  }
}

// CR lpuchallafiore: test evaluate Trunk
// CR lpuchallafiore: test evaluate Policy Head
// CR lpuchallafiore: test evaluate Value Head
