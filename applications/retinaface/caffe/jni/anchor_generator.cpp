#include "anchor_generator.h"
AnchorGenerator::AnchorGenerator() {
}

AnchorGenerator::~AnchorGenerator() {
}

// anchor plane
int AnchorGenerator::Generate(int fwidth, int fheight, int stride, float step, std::vector<int>& size, std::vector<float>& ratio, bool dense_anchor) {
    /*
    anchor_planes.resize(anchor_num);
    cv::Mat xs = cv::Mat(fheight, fwidth, CV_32FC1);
    cv::Mat ys = cv::Mat(fheight, fwidth, CV_32FC1);
    for (int w = 0; w < fwidth; ++w) {
        xs.col(w).setTo(float(w));
    }
    for (int h = 0; h < fheight; ++h) {
        ys.row(w).setTo(float(h));
    }
    xs = xs * stride;
    ys = ys * stride;
    for (int i = 0; i < anchor_num; ++i) {
        anchor_planes[i] = std::vector<cv::Mat>(4); 
        anchor_planes[i][0] = xs + anchors[i][0];
        anchor_planes[i][1] = ys + anchors[i][1];
        anchor_planes[i][2] = xs + anchors[i][2];
        anchor_planes[i][3] = ys + anchors[i][3];
    }
    */

    return 0;
}

// init different anchors
int AnchorGenerator::Init(int stride, const AnchorCfg& cfg, bool dense_anchor) {
	CRect2f base_anchor(0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1);
	std::vector<CRect2f> ratio_anchors;
	// get ratio anchors
	_ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
	_scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

	// save as x1,y1,x2,y2
	if (dense_anchor) {
		assert(stride % 2 == 0);
		int num = preset_anchors.size();
		for (int i = 0; i < num; ++i) {
			CRect2f anchor = preset_anchors[i];
			preset_anchors.push_back(CRect2f(anchor[0]+int(stride/2),
									anchor[1]+int(stride/2),
									anchor[2]+int(stride/2),
									anchor[3]+int(stride/2)));
		}
	}

    anchor_stride = stride;

	anchor_num = preset_anchors.size();

	return anchor_num;
}

int AnchorGenerator::FilterAnchor(MNN::Tensor& cls, MNN::Tensor& reg, MNN::Tensor& pts, std::vector<Anchor>& result) {
    assert(cls.channel() == anchor_num*2);
    assert(reg.channel() == anchor_num*4);
    int pts_length = 0;

	assert(pts.channel() % anchor_num == 0);
	pts_length = pts.channel() / anchor_num / 2;

    int w = cls.width();
    int h = cls.height();


    float* cls_data = cls.host<float>();
    float* reg_data = reg.host<float>();
    float* pts_data = pts.host<float>();

    for (int i = 0; i < h; ++i) 
    {
        for (int j = 0; j < w; ++j) 
        {
            for (int a = 0; a < anchor_num; ++a)
            {
                int idx = (anchor_num + a) * w * h + i * w + j;
                if (cls_data[idx] >= cls_threshold) 
                {
                    CRect2f box(
                        j * anchor_stride + preset_anchors[a][0],
                        i * anchor_stride + preset_anchors[a][1],
                        j * anchor_stride + preset_anchors[a][2],
                        i * anchor_stride + preset_anchors[a][3]
                    );
                    int box_idx0 = (a*4+0) * w * h + i * w + j;
                    int box_idx1 = (a*4+1) * w * h + i * w + j;
                    int box_idx2 = (a*4+2) * w * h + i * w + j;
                    int box_idx3 = (a*4+3) * w * h + i * w + j;
                    CRect2f delta(
                        reg_data[box_idx0],
                        reg_data[box_idx1],
                        reg_data[box_idx2],
                        reg_data[box_idx3]
                    );

                    Anchor res;
                    res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
                    bbox_pred(box, delta, res.finalbox);
                    res.score = cls_data[idx];
                    res.center = cv::Point(j,i);

                    if (1) {
                        std::vector<cv::Point2f> pts_delta(pts_length);
                        for (int p = 0; p < pts_length; ++p) {
                            int x_idx = (a*pts_length*2+p*2) * w * h + i * w + j;
                            int y_idx = (a*pts_length*2+p*2+1) * w * h + i * w + j;
                            pts_delta[p].x = pts_data[x_idx];
                            pts_delta[p].y = pts_data[y_idx];
                        }
                        landmark_pred(box, pts_delta, res.pts);
                    }
                    result.push_back(res);
                }
            }
        }
    }

    
	return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

	ratio_anchors.clear();
	float sz = w * h;
	for (int s = 0; s < ratios.size(); ++s) {
		float r = ratios[s];
		float size_ratios = sz / r;
		float ws = std::sqrt(size_ratios);
		float hs = ws * r;
		ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
								y_ctr - 0.5 * (hs - 1),
								x_ctr + 0.5 * (ws - 1),
								y_ctr + 0.5 * (hs - 1)));
	}
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
	scale_anchors.clear();
	for (int a = 0; a < ratio_anchor.size(); ++a) {
		CRect2f anchor = ratio_anchor[a];
		float w = anchor[2] - anchor[0] + 1;	
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		for (int s = 0; s < scales.size(); ++s) {
			float ws = w * scales[s];
			float hs = h * scales[s];
			scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
								y_ctr - 0.5 * (hs - 1),
								x_ctr + 0.5 * (ws - 1),
								y_ctr + 0.5 * (hs - 1)));
		}
	}

}

void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr; 
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w; 
    float pred_h = std::exp(dh) * h;

    box = cv::Rect_< float >(pred_ctr_x - 0.5 * (pred_w - 1.0),
            pred_ctr_y - 0.5 * (pred_h - 1.0),
            pred_ctr_x + 0.5 * (pred_w - 1.0),
            pred_ctr_y + 0.5 * (pred_h - 1.0));
}

void AnchorGenerator::landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) {
	float w = anchor[2] - anchor[0] + 1;	
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (int i = 0; i < delta.size(); ++i) {
        pts[i].x = delta[i].x*w + x_ctr;
        pts[i].y = delta[i].y*h + y_ctr;
    }
}
