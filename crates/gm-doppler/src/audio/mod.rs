//! From a media file to Doppler events: decode, decimate, spectrogram, ridges, fits.

pub mod comb;
pub mod decode;
pub mod detect;
pub mod events;
pub mod fit;
pub mod guide;
pub mod lines;
pub mod passlog;
pub mod pursuit;
pub mod resample;
pub mod ridges;
pub mod spectrogram;
pub mod templates;
pub mod tracker;

pub use decode::{Audio, AudioError, decode};
pub use detect::{DetectParams, Detection, Envelope, Pass, Rejected, detect};
pub use events::{
    Event, EventParams, SourceClass, Stage, discover, event_from_points, source_class,
};
pub use fit::{CvFit, fit_cv};
pub use passlog::{Curve, PassLog, PassRecord, UsedFor};
pub use resample::{ANALYSIS_RATE, decimate};
pub use ridges::{Ridge, RidgeParams, RidgePoint, ridges};
pub use spectrogram::{SpecParams, Spectrogram, spectrogram};
