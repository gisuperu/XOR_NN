#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

// 実行コマンド
// $clang++ xor_NN.cpp -std=c++11
//
// 活性化関数: ソフトプラス関数


std::random_device rd;
std::mt19937 gen(rd());
// unsigned seed = 0;
// std::mt19937 gen(seed);

struct Nuron{
    std::vector<double> weight;//0インデックス
    double outsig;      //h_{jp}
    double outsig_ac;   //g_{jp}
    double raw_gap;             //ε_{kp}
    std::vector<double> gap;    //∂J/∂weight
};


class NN{
private:
    int LAYER_SIZE;
    int IN_SIZE;
    int OUT_SIZE;
    double RHO; //学習率
    std::vector<std::vector<Nuron> > network;//0インデックス, 0インデックス(0はθ)
    std::vector<double> model_out; //1インデックス

    double active_func(double x);
    double active_dfunc(double x);
    void nuron_func(int layer, int idx, Nuron *nuron);
    void fix_func(int layer, int idx, Nuron *nuron);
    
    void wupdate(Nuron *nuron);
    bool readinput(std::vector<int> *input);
    bool readmodel(std::vector<int> *answer);

public:
    NN(int _layer, int _insize, int _outsize, double _rho);
    ~NN();
    void forward_propagation(std::vector<int> *indata);
    void practice(std::vector<int> *indata, std::vector<int> *answer);

    static double random_r(double low, double high);
    static int random_z(int low, int high);

    void show_out();
    void show_weight();
    void show_network();
};



//NN::public関数(コンストラクタとデコンストラクタ)
NN::NN(int _layersize, int _insize, int _outsize, double _rho){
    LAYER_SIZE = _layersize;
    IN_SIZE = _insize;
    OUT_SIZE = _outsize;
    RHO = _rho;
    
    //入力層の作成
    network.push_back(std::vector<Nuron>(IN_SIZE+1, Nuron{{}, 0, 0, 0, {}}));
    network[0][0].outsig = 1.0;
    network[0][0].outsig_ac = 1.0;

    //中間層を作成
    for(int i = 1/*入力層を引いている*/; i < LAYER_SIZE-1/*出力層を引いている*/; i++){
        network.push_back(std::vector<Nuron>(IN_SIZE+1, Nuron{{}, 0, 0, 0, {}})); //0番目のニューロンはθ=-wxを担う
        network[i][0].outsig = 1.0; 
        network[i][0].outsig_ac = 1.0;
        for(int j = 1; j <= IN_SIZE; j++){
            for(int k = 0; k <= IN_SIZE; k++){
                network[i][j].weight.push_back(random_r(-1.0, 1.0));
                // network[i][j].weight.push_back((j+k)%2 == 0 ? 1.0 : -1.0);//中間層のnuronごとに1，0に反応しやすくする[中間層の役割分担](事前学習法?)
                network[i][j].gap.push_back(0.0);
            }
        }
    }
    //出力層を作成
    network.push_back(std::vector<Nuron>(OUT_SIZE+1, Nuron{{}, 0, 0, 0, {}}));
    for(int j = 1; j <= OUT_SIZE; j++){
        for(int k = 0; k <= IN_SIZE; k++){
            network[LAYER_SIZE-1][j].weight.push_back(random_r(-1.0, 1.0));
            // network[LAYER_SIZE-1][j].weight.push_back((j+k)%2 == 0 ? 1.0 : -1.0);//中間層のnuronごとに1，0に反応しやすくする[中間層の役割分担](事前学習法?)
            network[LAYER_SIZE-1][j].gap.push_back(0.0);
        }
    }
    //出力の教師データ保存用vectorの確保
    for(int j = 0; j <= OUT_SIZE; j++){
        model_out.push_back(0.0);
    }


}

NN::~NN(){
}



//NN::private関数
double NN::active_func(double x){
    //ソフトプラス関数
    return std::log(1+std::exp(x));
}
double NN::active_dfunc(double x){
    //ソフトプラス関数
    double ex = std::exp(x);
    return ex/(1+ex);
}

/// @brief nuronの順伝搬計算関数()
/// @param layer 層の番号(0index)
/// @param idx 層内の番号[0はθ]
/// @param nuron 対象のnuron
void NN::nuron_func(int layer, int idx, Nuron *nuron){
    if(layer == 0 || idx == 0){
        return;
    }
    double insum = 0;
    for(int i = 0; i < network[layer-1].size(); i++){
        insum += nuron->weight[i] * network[layer-1][i].outsig_ac;
    }
    (*nuron).outsig = insum;
    (*nuron).outsig_ac = active_func(insum);
}

/// @brief nuronの誤差逆伝播の計算関数
/// @param layer 層の番号(0index)
/// @param idx 層内の番号[0はθ]
/// @param nuron 対象のnuron
void NN::fix_func(int layer, int idx, Nuron *nuron){
    if(layer == 0 || idx == 0){
        return;
    }
    double raw_gap = 0;
    double gap = 0;
    if(layer == LAYER_SIZE-1){
        //出力層の場合
        raw_gap = (nuron->outsig_ac - model_out[idx]) * active_dfunc(nuron->outsig);
        for(int i = 0; i < network[layer-1].size(); i++){
            gap = raw_gap * network[layer-1][i].outsig_ac;
            nuron->gap[i] = gap;
        }
        nuron->raw_gap = raw_gap;
    }else{
        //中間層の場合
        for(int i = 1; i < network[layer+1].size(); i++){//1から開始してθを除外
            // ∑εwの計算
            raw_gap += network[layer+1][i].raw_gap * network[layer+1][i].weight[idx];
        }
        raw_gap *= active_dfunc(nuron->outsig);
        for(int i = 0; i < network[layer-1].size(); i++){
            gap = raw_gap * network[layer-1][i].outsig_ac;
            nuron->gap[i] = gap;
        }
        nuron->raw_gap = raw_gap;
    }

    wupdate(nuron);
}

void NN::wupdate(Nuron *nuron){
    for(int i = 0; i < (*nuron).weight.size(); i++){
        nuron->weight[i] -= RHO * nuron->gap[i];
    }
}

bool NN::readinput(std::vector<int> *input){
    if(input->size() != IN_SIZE){
        return true;
    }
    int idx = 1;//1から開始してθを除外
    for(int in : (*input)){
        network[0][idx].outsig_ac = in;
        idx++;
    }
    return false;

}
bool NN::readmodel(std::vector<int> *answer){
    if(answer->size() != OUT_SIZE){
        return true;
    }
    int idx = 1;//1から開始してθを除外
    for(int ans : (*answer)){
        model_out[idx] = ans;
        idx++;
    }
    return false;
}




//NN::public関数
void NN::forward_propagation(std::vector<int> *indata){
    if(readinput(indata)){
        std::cerr << "入力の数が正しくありません" << std::endl;
        return;
    }
    for(int i = 1; i < LAYER_SIZE; i++){//1から始めて入力層の処理を飛ばす
        for(int j = 1; j < network[i].size(); j++){//1から始めてθを飛ばす
            nuron_func(i, j, &network[i][j]);
        }
    }
}
void NN::practice(std::vector<int> *indata, std::vector<int> *answer){
    forward_propagation(indata);
    if(readmodel(answer)){
        std::cerr << "出力モデルの数が正しくありません" << std::endl;
        return;
    }
    for(int i = LAYER_SIZE-1; i > 0; i--){//出力層から始めて入力層(0)を飛ばす
        for(int j = 1; j < network[i].size(); j++){//1から始めてθを飛ばす
            fix_func(i, j, &network[i][j]);
        }
    }
}

double NN::random_r(double low, double high){
    std::uniform_real_distribution<> randomDouble(low, high);
    return randomDouble(gen);
}
int NN::random_z(int low, int high){
    std::uniform_int_distribution<> randomInt(low, high);
    return randomInt(gen);
}

void NN::show_out(){
    // std::cout << "--出力--" << std::endl;
    for(int i = 1; i <= OUT_SIZE; i++){
        std::cout << std::fixed << std::setprecision(4) <<network[LAYER_SIZE-1][i].outsig_ac << std::endl;
    }
}
void NN::show_weight(){
    // std::cout << "--重み--" << std::endl;
    for(int i = 0; i < LAYER_SIZE; i++){
        std::cout << "- " << i << "層" << std::endl;
        for(Nuron nu : network[i]){
            for(double w : nu.weight){
                std::cout << std::fixed << std::setprecision(4) << w << ":";
            }
            std::cout << std::endl;
        }
    }
}
void NN::show_network(){
    std::cout << "network" << std::endl;
    std::cout << "layer: " << network.size() << std::endl;
    for(auto layer : network){
        std::cout << "\tnuron: " << layer.size() << std::endl;
        for(Nuron nuron : layer){
            std::cout << "\t\tweight   : " << nuron.weight.size();
            for(double w : nuron.weight){
                std::cout << std::fixed << std::setprecision(4) << ":" << w;
            }
            std::cout << std::endl;
            std::cout << "\t\toutsig   : " << nuron.outsig << std::endl;
            std::cout << "\t\toutsig_ac: " << nuron.outsig_ac << std::endl;
            std::cout << "\t\traw_gap  : " << nuron.raw_gap << std::endl;
            std::cout << "\t\tgap      : " << nuron.gap.size();
            for(double g : nuron.gap){
                std::cout << std::fixed << std::setprecision(4) << ":" << g;
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "model out: " << model_out.size() << std::endl;
}


// ============================== main関数 ==============================


int main(){
    std::vector<std::vector<int> > inset {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    std::vector<std::vector<int> > outset {
        {0},
        {1},
        {1},
        {0}
    };
    std::vector<std::vector<int> > pattern {
        {0,1,2,3},
        {0,1,3,2},
        {0,2,1,3},
        {0,2,3,1},
        {0,3,1,2},
        {0,3,2,1},
        {1,0,2,3},
        {1,0,3,2},
        {1,2,0,3},
        {1,2,3,0},
        {1,3,0,2},
        {1,3,2,0},
        {2,0,1,3},
        {2,0,3,1},
        {2,1,0,3},
        {2,1,3,0},
        {2,3,0,1},
        {2,3,1,0},
        {3,0,1,2},
        {3,0,2,1},
        {3,1,0,2},
        {3,1,2,0},
        {3,2,0,1},
        {3,2,1,0}
    };
    NN nnxor(3, 2, 1, 0.1);
    nnxor.show_weight();
    // nnxor.show_network();
    // nnxor.practice(&inset[1], &outset[1]);
    // nnxor.show_network();

    int training = 30000;
    // for(int i = 0; i < training/4; i++){//学習の偏りを無くすため全ての入力パターンが等確率にする
    //     int rand = NN::random_z(0,23);
    //     nnxor.practice(&inset[pattern[rand][0]], &outset[pattern[rand][0]]);
    //     nnxor.practice(&inset[pattern[rand][1]], &outset[pattern[rand][1]]);
    //     nnxor.practice(&inset[pattern[rand][2]], &outset[pattern[rand][2]]);
    //     nnxor.practice(&inset[pattern[rand][3]], &outset[pattern[rand][3]]);
    //     if((i*4)%5000 == 0){
    //         std::cout << i*4 << std::endl;
    //     }
    // }
    for(int i = 0; i < training; i++){//完全ランダムで学習データを選ぶ
        int rand = NN::random_z(0,3);
        nnxor.practice(&inset[rand], &outset[rand]);
        if(i%5000 == 0){
            std::cout << i << std::endl;
        }
    }

    //結果出力
    std::cout << "--学習結果--" << std::endl;
    for(int i = 0; i < 4; i++){
        double sum = 0;
        nnxor.forward_propagation(&inset[i]);
        std::cout << inset[i][0] << "," << inset[i][1] << ": ";
        nnxor.show_out();
    }
    nnxor.show_weight();
}
