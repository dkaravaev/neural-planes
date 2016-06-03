#include "Utils.hxx"
#include "SceneHandler.hxx"

// argv[1] = model name
// argv[2] = result name
// argv[3] = rotation x
// argv[4] = rotation y
// argv[5] = rotation z
// argv[6] = width
// argv[7] = height
int main(int argc, char** argv)
{
    srand(time(nullptr));

    auto model_name = std::string(argv[1]);
    auto result_name = std::string(argv[2]);

    auto x = std::stoi(argv[3]);
    auto y = std::stoi(argv[4]);
    auto z = std::stoi(argv[5]);

    auto width = static_cast<uint>(std::stoi(argv[6]));
    auto height = static_cast<uint>(std::stoi(argv[7]));

    SceneHandler::to_image(model_name, result_name, Utils::create_quat(x, y, z), width, height);

    return 0;
}




