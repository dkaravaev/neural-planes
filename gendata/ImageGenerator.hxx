//
// Created by dmitry on 6/11/16.
//

#ifndef MODEL_RENDER_GENERATORCONTEXT_HXX
#define MODEL_RENDER_GENERATORCONTEXT_HXX

#include <string>

#include <Magick++.h>
#include <json/json.h>

#include "ModelImage.hxx"
#include "SceneLoader.hxx"

class ImageGenerator
{
public:
    ImageGenerator(const Json::Value& config)
    {
        size.first = config["gendata"]["3dmodel"]["size"][0].asLargestUInt();
        size.second = config["gendata"]["3dmodel"]["size"][1].asLargestUInt();

        Json::Value rotation = config["gendata"]["3dmodel"]["rotation"];
        xrotation.first = rotation["x"][0].asInt();
        xrotation.second = rotation["x"][1].asInt();
        yrotation.first = rotation["y"][0].asInt();
        yrotation.second = rotation["y"][1].asInt();
        zrotation.first = rotation["z"][0].asInt();
        zrotation.second = rotation["z"][1].asInt();

        Json::Value clsmap = config["gendata"]["3dmodel"]["classmap"];

        blur = config["gendata"]["effects"]["blur"].asUInt();


        std::string backgrounds_folder = config["global"]["folders"]["backgrounds"].asString();
        std::string models_folder = config["global"]["folders"]["3dmodels"].asString();

        for (auto& background : config["global"]["files"]["backgrounds"])
        {
            backgrounds.push_back(backgrounds_folder + background.asString());
        }

        for (auto& model: config["global"]["files"]["3dmodels"])
        {
            models.push_back(models_folder + model.asString());
            classmap.push_back(clsmap[model.asString()].asString());
        }

        for (auto& cls : config["global"]["model"]["classes"])
        {
            classes.push_back(cls.asString());
        }
    }

    ImageContext create(const std::string& image_filename, const std::string& annotation_filename) const
    {
        ImageContext image(image_filename, annotation_filename);

        image.xrotation = xrotation.first + std::rand() % (xrotation.second - xrotation.first);
        image.yrotation = yrotation.first + std::rand() % (yrotation.second - yrotation.first);
        image.zrotation = zrotation.first + std::rand() % (zrotation.second - zrotation.first);

        ulong model_index = std::rand() % models.size();
        ulong background_index = std::rand() % backgrounds.size();
        image.model = model_index;
        image.background = background_index;

        image.width = size.first;
        image.height = size.second;

        image.cls = classmap[model_index];

        image.blur = blur;

        return image;
    }

public:
    std::vector<std::string> models;
    std::vector<Magick::Image> backgrounds;
    std::vector<std::string> classes;

    std::vector<std::string> classmap;

    uint blur;

    std::pair<int, int> xrotation;
    std::pair<int, int> yrotation;
    std::pair<int, int> zrotation;

    std::pair<std::size_t, std::size_t> size;
};


#endif //MODEL_RENDER_IMAGECONTEXT_HXX
