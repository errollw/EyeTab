#include "stdafx.h"

#include <memory>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <DShow.h>

#include <comet/comet.h>
#include <comet/util.h>
#include <comet/ptr.h>
#include <comet_task_ptr.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <format.h>

#include "GetGUID.h"

namespace comet {

#define MAKE_COMET_COMTYPE(T, BASE) \
    template<> struct comtype< ::T > \
    { \
        static const IID& uuid() { return IID_ ## T; } \
        typedef ::BASE base; \
    }

    MAKE_COMET_COMTYPE(IMF2DBuffer, IUnknown);
    MAKE_COMET_COMTYPE(IAMVideoProcAmp, IUnknown);
    MAKE_COMET_COMTYPE(IAMCameraControl, IUnknown);
}

namespace comet {
    template<typename TInf, typename TCount=UINT32>
    class com_ptr_array {
    public:
        com_ptr_array() : _ptr(nullptr), _count(0) {}
        ~com_ptr_array() {
            clear();
        }


        TInf** in() {
            return _ptr.in();
        }

        TInf*** inout() {
            return _ptr.inout();
        }

        TInf*** out() {
            clear();
            return _ptr.out();
        }

        TCount count() {
            return _count;
        }

        TCount* inout_count() {
            return &_count;
        }

        TCount* out_count() {
            clear();
            return &_count;
        }

        com_ptr<TInf> operator[](size_t i) {
            return com_ptr<TInf>(_ptr[i]);
        }



    private:
        void clear() {
            if (_ptr) {
                for (DWORD i = 0; i < _count; i++) {
                    _ptr[i]->Release();
                }
                _count = 0;
                _ptr.free();
            }
        }

        task_ptr<TInf*> _ptr;
        TCount _count;
    };
}

namespace comet
{
    struct auto_mf
    {
        auto_mf(DWORD dwFlags = MFSTARTUP_FULL)
        {
            MFStartup(MF_VERSION) | comet::raise_exception;
        }
        ~auto_mf()
        {
            MFShutdown();
        }
    };
}



float OffsetToFloat(const MFOffset& offset)
{
    return offset.value + (static_cast<float>(offset.fract) / 65536.0f);
}

void PrintAttributes( comet::com_ptr<IMFAttributes> attr )
{
    UINT32 count = 0;
    attr->GetCount(&count) | comet::raise_exception;
    if (count == 0) {
        std::cout << "Empty media type." << std::endl;
    }

    for (UINT32 j = 0; j < count; j++) {
        GUID guid;
        PROPVARIANT var;
        PropVariantInit(&var);
        attr->GetItemByIndex(j, &guid, &var);

        std::cout << GetGUIDName(guid) << " : ";

        if ((guid == MF_MT_FRAME_SIZE) || (guid == MF_MT_PIXEL_ASPECT_RATIO))
        {
            UINT32 uHigh = 0, uLow = 0;
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &uHigh, &uLow);
            std::cout << uHigh << " x " << uLow;
        }
        else if ((guid == MF_MT_FRAME_RATE) || (guid == MF_MT_FRAME_RATE_RANGE_MAX) ||
            (guid == MF_MT_FRAME_RATE_RANGE_MIN))
        {
            UINT32 uHigh = 0, uLow = 0;
            Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &uHigh, &uLow);
            std::cout << ((double)uHigh)/uLow;
        }
        else if ((guid == MF_MT_GEOMETRIC_APERTURE) || 
            (guid == MF_MT_MINIMUM_DISPLAY_APERTURE) || 
            (guid == MF_MT_PAN_SCAN_APERTURE))
        {
            MFVideoArea *pArea = reinterpret_cast<MFVideoArea*>(var.caub.pElems);
            std::cout << "(" << OffsetToFloat(pArea->OffsetX) << ", " << OffsetToFloat(pArea->OffsetY) << ") ";
            std::cout << "(" << pArea->Area.cx << ", " << pArea->Area.cy << ")";
        }
        else {
            switch (var.vt)
            {
            case VT_UI4:
                std::cout << var.ulVal;
                break;
            case VT_UI8:
                std::cout << var.uhVal.QuadPart;
                break;
            case VT_R8:
                std::cout << var.dblVal;
                break;
            case VT_CLSID:
                std::cout << GetGUIDName(*var.puuid);
                break;
            case VT_LPWSTR:
                std::cout << comet::bstr_t(var.pwszVal).s_str();
                break;
            case VT_VECTOR | VT_UI1:
                std::cout << "<<byte array>>";
                break;
            case VT_UNKNOWN:
                std::cout << "IUnknown";
                break;
            default:
                std::cout << "Unexpected attribute type (vt = " <<  var.vt << ")";
                break;
            }
        }
        PropVariantClear(&var);
        std::cout << std::endl;
    }
}


struct cam_prop_range
{
    long min;
    long max;
    long step;
    long defaultValue;
    long flags;
};
std::ostream& operator<<(std::ostream& os, const cam_prop_range& val) {
    os << "[" << val.min << ":" << val.step << ":" << val.max << "], " << val.defaultValue << ", " << val.flags;
    return os;
}

struct cam_prop_value
{
    long value;
    bool isAuto;

    cam_prop_value() : value(0), isAuto(true) {
    }
    cam_prop_value(long value) : value(value), isAuto(false) {
    }
    operator long() const {
        return value;
    }
};
std::ostream& operator<<(std::ostream& os, const cam_prop_value& val) {
    os << val.value;
    os << " (";
    if (val.isAuto)
        os << "auto";
    else
        os << "manual";
    os << ")";
    return os;
}

class media_type
{
public:
    media_type() : _ptr(NULL) {
    }
    media_type(IMFMediaType* ptr) : _ptr(ptr) {
    }
    media_type(comet::com_ptr<IMFMediaType> ptr) : _ptr(ptr) {
    }

    cv::Size resolution() const
    {
        UINT32 width, height;
        MFGetAttributeSize(_ptr.in(), MF_MT_FRAME_SIZE, &width, &height) | comet::raise_exception;
        return cv::Size(width, height);
    }
    double framerate() const
    {
        UINT32 num, denom;
        MFGetAttributeRatio(_ptr.in(), MF_MT_FRAME_RATE, &num, &denom) | comet::raise_exception;
        return static_cast<double>(num)/denom;
    }
    GUID subtype() const
    {
        GUID subtype;
        _ptr->GetGUID(MF_MT_SUBTYPE, &subtype) | comet::raise_exception;
        return subtype;
    }

    typedef media_type _Myt;
    _TYPEDEF_BOOL_TYPE;
    _OPERATOR_BOOL() const _NOEXCEPT
    {	// test for non-null pointer
        return (_ptr != 0 ? _CONVERTIBLE_TO_TRUE : 0);
    }

    comet::com_ptr<IMFMediaType> _ptr;
};


class camera {
public:
    camera(IMFActivate* ptr) : activate_ptr(ptr) {
    }
    camera(comet::com_ptr<IMFActivate> ptr) : activate_ptr(ptr) {
    }

    bool is_active() const {
        return !source_ptr.is_null();
    }
    void activate() {
        activate_ptr->ActivateObject(IID_IMFMediaSource, reinterpret_cast<void**>(source_ptr.out()));
        reader_ptr = NULL;
    }
    void shutdown() {
        activate_ptr->ShutdownObject();
        source_ptr = NULL;
        reader_ptr = NULL;
    }

    cv::Mat read_frame(int streamIndex = MF_SOURCE_READER_FIRST_VIDEO_STREAM, int bufferIndex = 0)
    {
        if (!reader_ptr) {
            comet::com_ptr<IMFAttributes> pAttributes;
            MFCreateAttributes(pAttributes.out(), 1) | comet::raise_exception;
            pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE) | comet::raise_exception;

            MFCreateSourceReaderFromMediaSource(source_ptr.in(), pAttributes.in(), reader_ptr.out()) | comet::raise_exception;  
        }

        comet::com_ptr<IMFSample> sample;
        DWORD actualStreamIndex, flags;
        LONGLONG timestamp;
        do {
            reader_ptr->ReadSample(
                streamIndex,        // Stream index.
                0,                  // Flags.
                &actualStreamIndex, // Receives the actual stream index.
                &flags,             // Receives status flags.
                &timestamp,         // Receives the time stamp.
                sample.out()        // Receives the sample or NULL.
                ) | comet::raise_exception;
        } while (sample == NULL && (flags & MF_SOURCE_READERF_STREAMTICK));

        media_type cur_media_type;
        reader_ptr->GetCurrentMediaType(actualStreamIndex, cur_media_type._ptr.out());

        cv::Mat ret(cur_media_type.resolution(), CV_8UC3);

        DWORD bufCount;
        sample->GetBufferCount(&bufCount) | comet::raise_exception;

        DWORD bufIndex = bufferIndex >= 0 ? bufferIndex : bufCount - bufferIndex;

        if (bufCount > bufIndex) {
            comet::com_ptr<IMFMediaBuffer> buffer;
            sample->GetBufferByIndex(bufferIndex, buffer.out()) | comet::raise_exception;

            comet::com_ptr<IMF2DBuffer> buf2d = try_cast(buffer);

            DWORD pcbLength;
            buf2d->GetContiguousLength(&pcbLength) | comet::raise_exception;

            COMET_ASSERT(ret.dataend - ret.datastart == pcbLength);

            struct buf_lock
            {
                comet::com_ptr<IMF2DBuffer>& buf2d;
                buf_lock(comet::com_ptr<IMF2DBuffer>& buf2d, BYTE*& scanline0, LONG& pitch) : buf2d(buf2d) {
                    buf2d->Lock2D(&scanline0, &pitch) | comet::raise_exception;
                }
                ~buf_lock() {
                    buf2d->Unlock2D() | comet::raise_exception;
                }
            };

            BYTE *scanline0;
            LONG pitch;
            buf_lock buf_lock_token(buf2d, scanline0, pitch);

            buf2d->ContiguousCopyTo(ret.data, DWORD(ret.dataend - ret.datastart)) | comet::raise_exception;
        }

        return ret;
    }

    std::string name() const {
        return get_attr_str(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
    }
    std::string symlink() const {
        return get_attr_str(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
    }

#define MAKE_CAMERA_PROPERTY(NAME, INTERFACE, PROPGUID) \
    bool has_ ## NAME() const \
    { \
        comet::com_ptr<IAM ## INTERFACE> pInterface = comet::com_cast(source_ptr); \
        if (!pInterface) return false; \
        long value, flags; \
        HRESULT hr = pInterface->Get(INTERFACE ## _ ## PROPGUID, &value, &flags); \
        return SUCCEEDED(hr); \
    } \
    cam_prop_range get_ ## NAME ## _range() const \
    { \
        comet::com_ptr<IAM ## INTERFACE> pInterface = comet::try_cast(source_ptr); \
        cam_prop_range range; \
        pInterface->GetRange(INTERFACE ## _ ## PROPGUID, &range.min, &range.max, &range.step, &range.defaultValue, &range.flags) | comet::raise_exception; \
        return range; \
    } \
    cam_prop_value get_ ## NAME() const \
    { \
        comet::com_ptr<IAM ## INTERFACE> pInterface = comet::try_cast(source_ptr); \
        cam_prop_value value; \
        long flags; \
        pInterface->Get(INTERFACE ## _ ## PROPGUID, &value.value, &flags) | comet::raise_exception; \
        value.isAuto = (flags & INTERFACE ## _Flags_Auto) != 0 || (flags & INTERFACE ## _Flags_Manual) == 0; \
        return value; \
    } \
    void set_ ## NAME(const cam_prop_value& value) const \
    { \
        comet::com_ptr<IAM ## INTERFACE> pInterface = comet::try_cast(source_ptr); \
        long flags = value.isAuto ? INTERFACE ## _Flags_Auto : INTERFACE ## _Flags_Manual; \
        pInterface->Set(INTERFACE ## _ ## PROPGUID, value.value, flags) | comet::raise_exception; \
    }

        MAKE_CAMERA_PROPERTY(exposure, CameraControl, Exposure)
        MAKE_CAMERA_PROPERTY(focus, CameraControl, Focus)
        MAKE_CAMERA_PROPERTY(zoom, CameraControl, Zoom)
        MAKE_CAMERA_PROPERTY(pan, CameraControl, Pan)
        MAKE_CAMERA_PROPERTY(tilt, CameraControl, Tilt)
        MAKE_CAMERA_PROPERTY(roll, CameraControl, Roll)
        MAKE_CAMERA_PROPERTY(iris, CameraControl, Iris)

        MAKE_CAMERA_PROPERTY(brightness, VideoProcAmp, Brightness)
        MAKE_CAMERA_PROPERTY(contrast, VideoProcAmp, Contrast)
        MAKE_CAMERA_PROPERTY(hue, VideoProcAmp, Hue)
        MAKE_CAMERA_PROPERTY(saturation, VideoProcAmp, Saturation)
        MAKE_CAMERA_PROPERTY(sharpness, VideoProcAmp, Sharpness)
        MAKE_CAMERA_PROPERTY(gamma, VideoProcAmp, Gamma)
        MAKE_CAMERA_PROPERTY(color_enable, VideoProcAmp, ColorEnable)
        MAKE_CAMERA_PROPERTY(white_balance, VideoProcAmp, WhiteBalance)
        MAKE_CAMERA_PROPERTY(backlight_compensation, VideoProcAmp, BacklightCompensation)
        MAKE_CAMERA_PROPERTY(gain, VideoProcAmp, Gain)

    std::vector<media_type> media_types(int streamIndex = 0) const {
        auto pHandler = getMediaTypeHandler(streamIndex);

        DWORD cTypes = 0;
        pHandler->GetMediaTypeCount(&cTypes) | comet::raise_exception;

        std::vector<media_type> ret;
        for (DWORD i = 0; i < cTypes; i++) {
            comet::com_ptr<IMFMediaType> pType;
            pHandler->GetMediaTypeByIndex(i, pType.out()) | comet::raise_exception;
            ret.emplace_back(pType);
        }

        return ret;
    }
    media_type get_media_type(int streamIndex = 0) {
        media_type ret;
        getMediaTypeHandler(streamIndex)->GetCurrentMediaType(ret._ptr.out());
        return ret;
    }
    void set_media_type(const media_type& type, int streamIndex = 0) {
        getMediaTypeHandler(streamIndex)->SetCurrentMediaType(type._ptr.in());
    }

private:
    std::string get_attr_str(REFGUID guid) const {
        comet::task_ptr<WCHAR> pStr;
        UINT32 strLen;
        activate_ptr->GetAllocatedString(guid, pStr.out(), &strLen) | comet::raise_exception;
        return comet::bstr_t(pStr.in(), strLen).s_str();
    }

    comet::com_ptr<IMFMediaTypeHandler> getMediaTypeHandler(int streamIndex = 0) const
    {
        comet::com_ptr<IMFPresentationDescriptor> pPD;
        source_ptr->CreatePresentationDescriptor(pPD.out()) | comet::raise_exception;

        BOOL fSelected;
        comet::com_ptr<IMFStreamDescriptor> pSD;
        pPD->GetStreamDescriptorByIndex(streamIndex, &fSelected, pSD.out()) | comet::raise_exception;

        comet::com_ptr<IMFMediaTypeHandler> pHandler;
        pSD->GetMediaTypeHandler(pHandler.out()) | comet::raise_exception;

        return pHandler;
    }

    comet::com_ptr<IMFActivate> activate_ptr;
    comet::com_ptr<IMFMediaSource> source_ptr;
    comet::com_ptr<IMFSourceReader> reader_ptr;
};

class camera_helper
{
public:
    static std::vector<camera> get_all_cameras()
    {
        comet::com_ptr<IMFAttributes> config;
        MFCreateAttributes(config.out(), 1) | comet::raise_exception;

        config->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID) | comet::raise_exception;

        comet::com_ptr_array<IMFActivate> com_ptr_array;
        MFEnumDeviceSources(config.in(), com_ptr_array.out(), com_ptr_array.out_count()) | comet::raise_exception;

        std::vector<camera> ret;
        for (size_t i = 0; i < com_ptr_array.count(); ++i) {
            ret.emplace_back(com_ptr_array[i]);
        }
        return ret;
    }
    static camera get_camera_by_symlink(const std::string& symlink)
    {
        // This is how you should do it, but for some reason it gives an activate pointer with no friendly name

//         comet::com_ptr<IMFAttributes> config;
//         MFCreateAttributes(config.out(), 1) | comet::raise_exception;
// 
//         config->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID) | comet::raise_exception;
//         comet::bstr_t symlink_bstr(symlink);
//         config->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, symlink_bstr.c_str()) | comet::raise_exception;
// 
//         comet::com_ptr<IMFActivate> activate_ptr;
//         MFCreateDeviceSourceActivate(config.in(), activate_ptr.out()) | comet::raise_exception;
// 
//         return camera(activate_ptr);

        for(auto&& camera : get_all_cameras())
        {
            if (camera.symlink() == symlink)
                return camera;
        }
        std::stringstream ss;
        ss << "No camera with symlink: " << symlink;
        throw std::runtime_error(ss.str());
    }
};


int _tmain(int argc, _TCHAR* argv[])
{
    comet::auto_coinit auto_coinit(COINIT_MULTITHREADED);
    comet::auto_mf auto_mf;
    
    //camera cam = camera_helper::get_camera_by_symlink("\\\\?\\usb#vid_046d&pid_082d&mi_00#7&f415157&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\{bbefb6c7-2fc4-4139-bb8b-a58bba724083}");
    for (auto&& cam : camera_helper::get_all_cameras())
    {
        std::cout << cam.name() << std::endl;
        std::cout << cam.symlink() << std::endl;

        cam.activate();

#define MAP_OVER_PROPERTIES(FUNC) \
    FUNC(exposure) \
    FUNC(focus) \
    FUNC(zoom) \
    FUNC(pan) \
    FUNC(tilt) \
    FUNC(roll) \
    FUNC(iris) \
    FUNC(brightness) \
    FUNC(contrast) \
    FUNC(hue) \
    FUNC(saturation) \
    FUNC(sharpness) \
    FUNC(gamma) \
    FUNC(color_enable) \
    FUNC(white_balance) \
    FUNC(backlight_compensation) \
    FUNC(gain)

#define INIT_PROPERTY(PROPERTY) \
    bool has_ ## PROPERTY = cam.has_ ## PROPERTY(); \
    cam_prop_range PROPERTY ## _range; \
    if (has_ ## PROPERTY) { \
        PROPERTY ## _range = cam.get_ ## PROPERTY ## _range(); \
        std::cout << #PROPERTY " range: " << PROPERTY ## _range << std::endl; \
    }

        MAP_OVER_PROPERTIES(INIT_PROPERTY)

        media_type bestType;

        for (auto&& media_type : cam.media_types()) {
            if (media_type.subtype() == MFVideoFormat_RGB24) {
                double framerate = media_type.framerate();
                cv::Size resolution = media_type.resolution();

                std::cout << resolution << " " << framerate << std::endl;

                if (!bestType || 
                    (
                        (framerate > bestType.framerate())|| 
                        (framerate == bestType.framerate() && resolution.area() > bestType.resolution().area())
                    ) && resolution.height < 800
                ) {
                    bestType = media_type;
                }
            }
        }

        cam.set_media_type(bestType);
        PrintAttributes(cam.get_media_type()._ptr);

        cv::namedWindow("m");

#define INIT_PROPERTYSLIDER(PROPERTY) \
    int PROPERTY ## _slider; \
    if (has_ ## PROPERTY) { \
        cv::createTrackbar(#PROPERTY, "m", &PROPERTY ## _slider, (PROPERTY ## _range.max - PROPERTY ## _range.min) / PROPERTY ## _range.step); \
        PROPERTY ## _slider = (PROPERTY ## _range.defaultValue - PROPERTY ## _range.min) / PROPERTY ## _range.step; \
        cv::setTrackbarPos(#PROPERTY, "m", PROPERTY ## _slider); \
    }

        MAP_OVER_PROPERTIES(INIT_PROPERTYSLIDER)

        while(true) {
            #define USE_PROPERTYSLIDER(PROPERTY) \
                try { \
                    if (has_ ## PROPERTY) { \
                        cam.set_ ## PROPERTY(PROPERTY ## _slider * PROPERTY ## _range.step + PROPERTY ## _range.min); \
                    } \
                } catch (comet::com_error& e) { \
                    std::cerr << "Error setting " #PROPERTY ": " << e.what() << std::endl; \
                }

            MAP_OVER_PROPERTIES(USE_PROPERTYSLIDER)

            cv::Mat m = cam.read_frame();
            cv::imshow("m", m);
            if (cv::waitKey(1) != -1){
                break;
            }
        }
    }

	return 0;
}

