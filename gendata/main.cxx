#include <fstream>

#include <json/json.h>

#include "Utils.hxx"
#include "SceneLoader.hxx"

void generate_data(int number, const std::string& images_folder, const std::string& annotations_folder,
                   const ImageGenerator& generator)
{
    for (int i = 0; i < number; ++i)
    {
        std::string image_filename = images_folder + std::to_string(i) + ".png";
        std::string annotation_filename = annotations_folder + std::to_string(i) + ".xml";

        ImageContext image = generator.create(image_filename, annotation_filename);
        SceneLoader::create(image, &generator);
    }
}


int main(int argc, char** argv)
{
    std::srand(time(nullptr));
    Magick::InitializeMagick(*argv);

    auto filename = std::string(argv[1]);
    std::ifstream fin(filename);

    Json::Reader reader;
    Json::Value config;
    bool success = reader.parse(fin, config);

    if (!success)
    {
        std::cerr  << "Failed to parse configuration\n"  << reader.getFormattedErrorMessages();
        return EXIT_FAILURE;
    }

    ImageGenerator generator(config);

    int train_num = config["gendata"]["number"]["train"].asInt();
    int test_num = config["gendata"]["number"]["test"].asInt();
    int validation_num = config["gendata"]["number"]["validation"].asInt();

    fclose(stderr);
    fclose(stdout);

    generate_data(train_num, config["global"]["folders"]["images"]["train"].asString(),
                  config["global"]["folders"]["annotations"]["train"].asString(), generator);

    generate_data(test_num, config["global"]["folders"]["images"]["test"].asString(),
                  config["global"]["folders"]["annotations"]["test"].asString(), generator);

    generate_data(validation_num, config["global"]["folders"]["images"]["validation"].asString(),
                  config["global"]["folders"]["annotations"]["validation"].asString(), generator);

    return EXIT_SUCCESS;
}




