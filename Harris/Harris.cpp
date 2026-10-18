#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
struct XYGrad{
    cv::Mat grad_x;
    cv::Mat grad_y;
};

struct TensorM{
    cv::Mat A;
    cv::Mat B;
    cv::Mat C;
};

struct KeyPoint{
    int x;
    int y;
    float score;
};
// 读取彩色图片，并将其转换为灰度图像
cv::Mat readImg(std::string imgPath){ 
    cv::Mat grayImg = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    // 这里需要补一下检查为空的情况;
    return grayImg; 
}

void Gaussian(const cv::Mat& img, cv::Mat& Blurred, double sigma = 1.0){
    int ksize = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
    cv::GaussianBlur(img, Blurred, cv::Size(ksize,ksize), sigma);
   }
// 计算水平梯度与垂直梯度
void calGrad(const cv::Mat& img, XYGrad& Grad){ 
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);
    Grad.grad_x = grad_x;
    Grad.grad_y = grad_y;
}

// 计算响应值所构成的矩阵
void getHarrisResponse(const TensorM& M, cv::Mat& response){
    int rows = M.A.rows;
    int cols = M.A.cols;
    for (int y=0; y<rows; y++){
        for (int x=0; x<cols; x++){
            float a = M.A.at<float>(y, x);
            float b = M.B.at<float>(y, x);
            float c = M.C.at<float>(y, x);
            float R = a * b - c * c - 0.04f * (a + b) * (a + b);
            response.at<float>(y, x) = R;
        }
    } 
}


bool isLocalMaximum(const cv::Mat& neighborhood){
    double maxVal;
    cv::minMaxLoc(neighborhood, nullptr, &maxVal);
    int rows = neighborhood.rows;
    int cols = neighborhood.cols;
    int cx = cols / 2;
    int cy = rows / 2; 
    return neighborhood.at<float>(cy, cx) >= maxVal && neighborhood.at<float>(cy, cx) > 0; 
}


cv::Mat NMS(const cv::Mat& response, int nonMaxSize = 5){
    cv::Mat result(response.size(), CV_32F, cv::Scalar(0));
    int safe_range = nonMaxSize / 2;
    for (int y=safe_range; y<response.rows-safe_range; y++){
        for (int x=safe_range; x<response.cols-safe_range; x++){
            cv::Rect roi(x-safe_range, y-safe_range, nonMaxSize, nonMaxSize);
            cv::Mat neighbour = response(roi);
            if(isLocalMaximum(neighbour)){
                result.at<float>(y, x) = response.at<float>(y, x);
            }
        }
    }
    return result;
}



bool compareByScore(const KeyPoint& a, const KeyPoint& b){
    return a.score > b.score;
}

std::vector<KeyPoint>getkeypoints(const cv::Mat& keymap, double nonMaxvalue, int nFeature = -1){
    std::vector<KeyPoint> kps;
    // 找出所有不等于 nonMaxvalue 的部分，也就是最大值
    for (int y=0;y<keymap.rows;y++){
        for (int x=0;x<keymap.cols;x++){
           if (keymap.at<float>(y, x)!=nonMaxvalue){
                kps.push_back({x, y, keymap.at<float>(y, x)});
           } 
        }
    }
    std::cout<<kps.size()<<" keypoints were found "<<std::endl;
    // 如果对数量有要求，就优先选取那些比较大的数值
    if (nFeature!=-1 && nFeature<kps.size()){
        std::sort(kps.begin(), kps.end(), compareByScore); // 这里使用 C++ 中的排序函数，从大到小进行排序
        kps.resize(nFeature); // 直接截断了，把后面值比较低的舍去
    }

    return kps;

}

// 可视化，将关键点画在图中
cv::Mat drawKeyPoints(const cv::Mat& image, const std::vector<KeyPoint>& kps){
    cv::Mat keyPointsImg = image;
    for (const auto& kp: kps){
        cv::circle(keyPointsImg, cv::Point(kp.x, kp.y), 3, cv::Scalar(0, 0, 255)); // 将关键点画在图中
    }
    return keyPointsImg;
}

int main(){
    // 读取 RGB 图像并变成灰度图
    cv::Mat img;
    std::string imgPath = "img.jpg";
    img = readImg(imgPath);

    // 先高斯模糊，然后计算水平方向和垂直方向的梯度
    cv::Mat Blurred;
    Gaussian(img, Blurred);
    XYGrad Grad;
    calGrad(Blurred, Grad);
    // 对梯度图进行乘积
    TensorM M;
    cv::multiply(Grad.grad_x, Grad.grad_x, M.A);
    cv::multiply(Grad.grad_y, Grad.grad_y, M.B);
    cv::multiply(Grad.grad_x, Grad.grad_y, M.C);
    // 对乘积后的梯度图进行高斯模糊，已经提前完成了加权求和，现在只需把 Tensor 中对应的元素取出来就行了
    Gaussian(M.A, M.A);
    Gaussian(M.B, M.B);
    Gaussian(M.C, M.C);
    // 计算响应值所构成的矩阵
    cv::Mat response(M.A.size(), M.A.type(), cv::Scalar(0));
    getHarrisResponse(M, response);

    // 利用阈值过滤，矩阵中元素 > 0 才能是角点的候选
    response.setTo(0, response <= 0);
    // NMS
    cv::Mat result = NMS(response);

    // 得到角点
    std::vector<KeyPoint> kps = getkeypoints(result, 0, 300); // 这里可以对参数做修改，-1 代表取了所有的角点
    
    // 可视化
    cv::Mat colorCanvas = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::Mat img_kps = drawKeyPoints(colorCanvas, kps);
    cv::imwrite("Harris.png", img_kps);
    return 0;
}