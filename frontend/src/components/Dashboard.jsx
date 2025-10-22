import React, { useState, useEffect } from 'react';
import { collection, query, where, orderBy, onSnapshot } from 'firebase/firestore';
import { db } from '../firebase/config';
import { useAuth } from '../contexts/AuthContext';
import Card from './ui/card';
import PrimaryRoundedButton from './ui/button';

const Dashboard = () => {
  const [uploads, setUploads] = useState([]);
  const [loading, setLoading] = useState(true);
  const { currentUser } = useAuth();

  useEffect(() => {
    if (!currentUser) return;

    const q = query(
      collection(db, 'model_uploads'),
      where('userId', '==', currentUser.uid),
      orderBy('createdAt', 'desc')
    );

    const unsubscribe = onSnapshot(q, (querySnapshot) => {
      const uploadsData = [];
      querySnapshot.forEach((doc) => {
        uploadsData.push({ id: doc.id, ...doc.data() });
      });
      setUploads(uploadsData);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [currentUser]);

  const handleDownload = (filename) => {
    // Get the auth token for the request
    currentUser.getIdToken().then((token) => {
      window.location.href = `/api/download/${filename}?token=${token}`;
    });
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-64">
        <div className="text-lg">Loading your uploads...</div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6 text-blue-900">Your Model Uploads</h2>
      
      {uploads.length === 0 ? (
        <Card
          title="No uploads yet"
          description="Upload and optimize your first model to see it here!"
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {uploads.map((upload) => (
            <Card
              key={upload.id}
              title={upload.originalFilename}
              description={
                <div className="space-y-2">
                  <div className="text-sm text-gray-600">
                    <strong>Target Device:</strong> {upload.targetDevice}
                  </div>
                  <div className="text-sm text-gray-600">
                    <strong>Uploaded:</strong> {new Date(upload.createdAt.toDate()).toLocaleDateString()}
                  </div>
                  {upload.metrics && (
                    <div className="text-sm space-y-1">
                      <div><strong>Original Size:</strong> {upload.metrics.original_size_mb?.toFixed(2)} MB</div>
                      <div><strong>Optimized Size:</strong> {upload.metrics.optimized_size_mb?.toFixed(2)} MB</div>
                      <div><strong>Size Reduction:</strong> {upload.metrics.size_reduction_percent?.toFixed(1)}%</div>
                      <div><strong>Latency Improvement:</strong> {upload.metrics.original_latency_ms && upload.metrics.optimized_latency_ms ? 
                        ((upload.metrics.original_latency_ms - upload.metrics.optimized_latency_ms) / upload.metrics.original_latency_ms * 100).toFixed(1) + '%' : 'N/A'}</div>
                    </div>
                  )}
                  {upload.optimizedFilename && (
                    <div className="pt-2">
                      <PrimaryRoundedButton
                        onClick={() => handleDownload(upload.optimizedFilename)}
                        className="w-full text-sm"
                      >
                        Download Optimized Model
                      </PrimaryRoundedButton>
                    </div>
                  )}
                </div>
              }
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
