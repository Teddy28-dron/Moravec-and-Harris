// 代码参考了 https://zhaoxuhui.top/blog/2019/01/10/MoravecCornerDetection.html，在此对作者表示感谢
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

struct KeyPoint{
    int x;
    int y;
    float score;
};

struct window{
    cv::Mat win;
    int start_x, end_x;
    int start_y, end_y;
};
// 计算差异
double calcV(cv::Mat windows1, cv::Mat windows2){
    cv::Mat diff;
    cv::subtract(windows1, windows2, diff, cv::noArray(), CV_32F); // 这里需要转成 32 位的模式，否则可能会出现数值溢出的情况
    cv::multiply(diff, diff, diff);
    return cv::sum(diff)[0]; // 注意这里返回的是个 double 
}

// 获得指定大小窗口的范围和内容
// 实际上就是截取一部分的内容
// 这里其实是一个超参数
cv::Mat getWindow(cv::Mat image, int i, int j, int win_size){
    if (win_size % 2 == 0){
        return cv::Mat(); // 窗口大小必须是奇数，否则返回空矩阵
    }
   int half_size = win_size / 2;
   int start_x = j - half_size;
   int start_y = i - half_size;

   cv::Rect roi(start_x, start_y, win_size, win_size);
   return image(roi);
}

// 把指定位置的窗口剪裁出来
struct window getWindowWithRange(cv::Mat img, int i, int j, int win_size){
    if (win_size%2 == 0){
        // ?
    }
    window winW;
    int half_size = win_size / 2;
    int start_x = j - half_size;
    int start_y = i - half_size;
    int end_x = j + half_size;
    int end_y = i + half_size;
    cv::Rect region(start_x, start_y, win_size, win_size);
    winW.win = img(region);
    winW.start_x = start_x;
    winW.start_y = start_y;
    winW.end_x = end_x;
    winW.end_y = end_y;
    return winW;  
}

cv::Mat getDirectionWin(cv::Mat image, int i, int j, int xoffset, int yoffset, int win_size){
    int half_size = win_size / 2; 
    cv::Rect Droi(j-half_size+xoffset, i-half_size+yoffset, win_size, win_size);
    return image(Droi); 
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

bool compareByScore(const KeyPoint& a, const KeyPoint& b){
    return a.score > b.score;
}

// 
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


// 将整个函数串起来
std::tuple<std::vector<KeyPoint>, cv::Mat> getMoravecKps(std::string img_path, int win_size = 3, int win_offset = 1, int nonMaxSize = 5, int nonMaxValue = 0, int nFeature = -1, int thCRF = -1){
    // 参数解释：
    // img_path: 影像路径
    // win_size: 滑动窗口的大小
    // win_offset: 偏移量
    // nonMaxSize: 非极大值抑制时的窗口大小
    // nonMaxValue: 非极大值抑制时，非极大值所取的值
    // nFeature: 打算提取的个数，-1表示保留所有提取出来的极大值
    // thCRF: 在对CRF进行筛选时使用的阈值，-1表示自动计算平均值作为阈值
    std::cout<<"step1: read image"<<std::endl;
    cv::Mat image = cv::imread(img_path, cv::IMREAD_GRAYSCALE); // 直接就将灰度图像读出来（虽然原始图像是彩色的）


    int img_h = image.rows;
    int img_w = image.cols;

    cv::Mat keymap(image.size(), CV_32F, cv::Scalar(0));

    std::cout<<"=>image size"<<img_h<<"*"<<img_w<<std::endl;

    std::cout<<"step2: calculate score value using sliding window"<<std::endl; int safe_range = win_offset + win_size;

    // top<=>bottom left<=>right 
    
    cv::Mat win_t, win_rt, win_r, win_rb, win_b, win_lb, win_l, win_lt;
    for (int i=safe_range; i<img_h-safe_range; i++){
        for(int j=safe_range; j<img_w-safe_range; j++){
            // 对于每一个点，我们都要计算 8 个方向上的梯度的平方和，并将其最小值填入到 keymap 中
            // 首先找到原始的窗口：
            cv::Mat win = getWindow(image, i, j, win_size);
            win_t =  getDirectionWin(image, i, j, 0, -win_offset, win_size);
            win_rt =  getDirectionWin(image, i, j, win_offset, -win_offset, win_size);
            win_r =  getDirectionWin(image, i, j, win_offset, 0, win_size);
            win_rb =  getDirectionWin(image, i, j, win_offset, win_offset, win_size);
            win_b =  getDirectionWin(image, i, j, 0, win_offset, win_size);
            win_lb =  getDirectionWin(image, i, j, -win_offset, win_offset, win_size);
            win_l =  getDirectionWin(image, i, j, -win_offset, 0, win_size);
            win_lt =  getDirectionWin(image, i, j, -win_offset, -win_offset, win_size);
            double v1 = calcV(win, win_t);
            double v2 = calcV(win, win_rt);
            double v3 = calcV(win, win_r);
            double v4 = calcV(win, win_rb);
            double v5 = calcV(win, win_b);
            double v6 = calcV(win, win_lb);
            double v7 = calcV(win, win_l);
            double v8 = calcV(win, win_lt);

            double minValue  = std::min({v1, v2, v3, v4, v5, v6, v7, v8});
            keymap.at<float>(i, j) = minValue;
        }

    } 
    float mean_c;
    if (thCRF == -1){
        cv::Scalar meanVal = cv::mean(keymap);
        mean_c = meanVal.val[0]; 
        std::cout<<"=>auto threshold for score value: "<<mean_c;
    }
    else{
        mean_c = thCRF;
        std::cout<<"=>threshold for score value: "<<mean_c;
    }

    std::cout<<"step 3:filter keypoints using threshold..."<<std::endl;
    cv::imwrite("keymap.jpg", keymap);  
    keymap.setTo(0, keymap < mean_c);
    cv::imwrite("keymap_th.jpg", keymap);

    std::cout<<"step 4:non maximum supression..."<<std::endl;
    window winW;
    cv::Mat result(keymap.size(), keymap.type(), cv::Scalar(nonMaxValue));
    for (int y=safe_range; y<img_h-safe_range; y++){
        for(int x=safe_range; x<img_w-safe_range; x++){
            winW = getWindowWithRange(keymap, y, x, nonMaxSize);
            cv::Rect roi(winW.start_x, winW.start_y, winW.end_x - winW.start_x + 1, winW.end_y - winW.start_y + 1);
            if(isLocalMaximum(keymap(roi))){
               result.at<float>(y, x) = keymap.at<float>(y, x); 
            };
        }
    }

    std::cout<<"step 5:get keypoint location and draw points"<<std::endl;
    cv::Mat colorCanvas = cv::imread(img_path, cv::IMREAD_COLOR);
    std::vector<KeyPoint> kps = getkeypoints(result, nonMaxValue, nFeature);
    cv::Mat img_kps = drawKeyPoints(colorCanvas, kps);
    return {kps, img_kps};

}

int main(){
    auto [corners, result] = getMoravecKps("img1.jpg", 3, 1, 5, 0, 300, -1);
    cv::imwrite("moravec1.jpg", result);
    return 0;
}
