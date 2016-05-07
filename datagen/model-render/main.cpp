#include <sstream>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgUtil/Optimizer>
#include <osgViewer/Viewer>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osg/AutoTransform>


class SnapImageDrawCallback : public osg::Camera::DrawCallback
{
public:

    SnapImageDrawCallback(const std::string& filename)
            : filename(filename)
    {

    }

    virtual void operator ()(const osg::Camera& camera) const
    {
        int x, y;
        int width,height;
        x = static_cast<int>(camera.getViewport()->x());
        y = static_cast<int>(camera.getViewport()->y());
        width = static_cast<int>(camera.getViewport()->width());
        height = static_cast<int>(camera.getViewport()->height());

        osg::ref_ptr<osg::Image> image = new osg::Image;
        image->readPixels(x,y,width,height,GL_RGB,GL_UNSIGNED_BYTE);

        osgDB::writeImageFile(*image, filename);
    }

protected:
    std::string filename;
};

void scene_to_image(osg::Node* model, const std::string& filename, const osg::Quat& rotation, uint width, uint height)
{
    osgViewer::Viewer viewer;

    osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keySwitchManipulator = new osgGA::KeySwitchMatrixManipulator;
    keySwitchManipulator->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator());
    viewer.setCameraManipulator(keySwitchManipulator.get());

    osg::ref_ptr<osg::GraphicsContext> pbuffer;

    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
    {
        traits->x = 0;
        traits->y = 0;
        traits->width = width;
        traits->height = height;
        traits->red = 8;
        traits->green = 8;
        traits->blue = 8;
        traits->alpha = 8;
        traits->windowDecoration = false;
        traits->pbuffer = true;
        traits->doubleBuffer = true;
        traits->sharedContext = 0;
    }

    pbuffer = osg::GraphicsContext::createGraphicsContext(traits.get());

    osgUtil::Optimizer optimizer;
    optimizer.optimize(model);

    osg::AutoTransform* trans = new osg::AutoTransform;
    trans->addChild(model);
    trans->setRotation(rotation);

    viewer.setSceneData(trans);
    viewer.getCamera()->setClearColor(osg::Vec4(1,1,1,1));

    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setGraphicsContext(pbuffer.get());
    camera->setViewport(new osg::Viewport(0,0,width,height));

    GLenum buffer = pbuffer->getTraits()->doubleBuffer ? GL_BACK : GL_FRONT;
    camera->setDrawBuffer(buffer);
    camera->setReadBuffer(buffer);
    camera->setFinalDrawCallback(new SnapImageDrawCallback(filename));

    viewer.addSlave(camera.get(), osg::Matrixd(), osg::Matrixd());

    viewer.realize();

    viewer.frame();
}


osg::Quat parse_quat(const std::string& str)
{
    std::istringstream iss(str);
    std::string sub;
    iss >> sub;
    double a = std::stod(sub);
    iss >> sub;
    double b = std::stod(sub);
    iss >> sub;
    double c = std::stod(sub);
    iss >> sub;
    double d = std::stod(sub);

    return osg::Quat(a, b, c, d);
}

// argv[1] = model name
// argv[2] = result name
// argv[3] = rotation string
// argv[4] = width
// argv[5] = height
int main(int argc, char** argv)
{
    auto model_name = std::string(argv[1]);
    auto result_name = std::string(argv[2]);
    auto rotation = std::string(argv[3]);

    auto width = static_cast<uint>(std::stoi(argv[4]));
    auto height = static_cast<uint>(std::stoi(argv[5]));

    osg::Node* model = osgDB::readNodeFile(model_name);
    scene_to_image(model, result_name, parse_quat(rotation), width, height);

    return 0;
}




