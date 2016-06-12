//
// Created by dmitry on 5/8/16.
//

#ifndef MODEL_RENDER_SCENEHANDLER_HXX
#define MODEL_RENDER_SCENEHANDLER_HXX

#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osg/AutoTransform>
#include <osgViewer/Viewer>
#include <osg/Node>

#include <Magick++.h>
#include "Utils.hxx"
#include "ModelImage.hxx"
#include "ImageGenerator.hxx"


class ImageDrawCallback : public osg::Camera::DrawCallback
{
public:
    ImageDrawCallback(const ImageContext& context, const ImageGenerator* generator)
        : context(context), generator(generator)
    {

    }


    virtual void operator()(const osg::Camera &camera) const
    {
        int x, y;
        int width, height;
        x = static_cast<int>(camera.getViewport()->x());
        y = static_cast<int>(camera.getViewport()->y());
        width = static_cast<int>(camera.getViewport()->width());
        height = static_cast<int>(camera.getViewport()->height());

        osg::ref_ptr<osg::Image> sceneshot = new osg::Image;
        sceneshot->readPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE);

        Magick::Image obj = Magick::Image(static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                                          "RGB", Magick::StorageType::CharPixel, sceneshot->data());

        Magick::Image background = generator->backgrounds[context.background];
        Magick::ColorRGB color = obj.pixelColor(0, 0);

        obj.transparent(color);
        obj.trim();

        int xpos = int(std::rand() % (background.columns() - obj.columns()));
        int ypos = int(std::rand() % (background.rows() - obj.rows()));


        background.composite(obj, xpos, ypos, Magick::DissolveCompositeOp);
        background.blur(context.blur);
        background.write(context.image_filename);

        context.to_xml(xpos, ypos, int(xpos + obj.columns()), int(ypos + obj.rows()));
    }

public:
    ImageContext context;
    const ImageGenerator* generator;
};


class SceneLoader
{
public:
    static void create(const ImageContext& context, const ImageGenerator* generator)
    {
        osgViewer::Viewer viewer;

        osg::ref_ptr<osgGA::KeySwitchMatrixManipulator> keySwitchManipulator = new osgGA::KeySwitchMatrixManipulator;
        keySwitchManipulator->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator());
        viewer.setCameraManipulator(keySwitchManipulator.get());

        osg::ref_ptr<osg::GraphicsContext> pbuffer;

        int width = int(context.width);
        int height = int(context.height);
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

        osg::Node* model = osgDB::readNodeFile(generator->models[context.model]);

        osgUtil::Optimizer optimizer;
        optimizer.optimize(model);

        osg::AutoTransform* trans = new osg::AutoTransform;
        trans->addChild(model);
        trans->setRotation(Utils::create_quat(context.xrotation, context.yrotation, context.zrotation));

        viewer.setSceneData(trans);
        viewer.getCamera()->setClearColor(osg::Vec4(1, 1, 1, 1));

        osg::ref_ptr<osg::Camera> camera = new osg::Camera;
        camera->setGraphicsContext(pbuffer.get());
        camera->setViewport(new osg::Viewport(0, 0, width, height));

        GLenum buffer = pbuffer->getTraits()->doubleBuffer ? GL_BACK : GL_FRONT;
        camera->setDrawBuffer(buffer);
        camera->setReadBuffer(buffer);

        ImageDrawCallback* callback = new ImageDrawCallback(context, generator);
        camera->setFinalDrawCallback(callback);

        viewer.addSlave(camera.get(), osg::Matrixd(), osg::Matrixd());

        viewer.realize();

        viewer.frame();
    }
};

#endif //MODEL_RENDER_SCENEHANDLER_HXX
